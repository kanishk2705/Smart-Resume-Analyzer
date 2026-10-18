"""Scores every pair in the benchmark with each method and reports how well each ranks them.

    python -m evaluation.run

Writes evaluation/results/results.json and evaluation/results/RESULTS.md.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from core.config import Settings
from core.scoring import ResumeAnalyzer, SentenceTransformerEncoder
from evaluation.data import DATASET, load_split, resume_fold
from evaluation.features import CachingEncoder, DocumentIndex, embedding_features, pair_features
from evaluation.finetune import FINE_TUNED_SCORES, UNSEEN_FOLDS, UNSEEN_SCORES
from evaluation.metrics import (
    auc_all_pairs,
    auc_within_job,
    job_bootstrap,
    pair_bootstrap,
    per_job_auc,
    spearman,
)
from evaluation.models import (
    PairwiseRanker,
    cross_validated_scores,
    pointwise_boosting,
    pointwise_logistic,
)

logger = logging.getLogger("evaluation")

HERE = Path(__file__).parent
COMPONENTS = ["skills", "coverage", "experience", "education"]

# Methods that need no training data
FIXED_METHODS = {
    "tfidf": "TF-IDF cosine (lexical baseline)",
    "v1_whole_document": "v1: whole-document SBERT cosine",
    "skills": "v2 component: skills match",
    "coverage": "v2 component: requirement coverage",
    "experience": "v2 component: experience relevance",
    "education": "v2 component: education",
    "v2_overall": "v2 overall (hand-set weights)",
    "resume_prior": "Leakage check: resume's training labels, job ignored",
}
# Methods fitted on the training split: label, feature set, model factory
LEARNED_METHODS = {
    "components_logistic": (
        "Logistic regression on the 4 v2 components",
        "components",
        pointwise_logistic,
    ),
    "interaction_logistic": (
        "Logistic regression on interaction features",
        "interaction",
        pointwise_logistic,
    ),
    "interaction_boosting": (
        "Gradient boosting on interaction features",
        "interaction",
        pointwise_boosting,
    ),
    "interaction_ranker": (
        "Pairwise ranker on interaction features",
        "interaction",
        PairwiseRanker,
    ),
    "embedding_ranker": (
        "Pairwise ranker on pooled embeddings",
        "embedding",
        lambda: PairwiseRanker(regularisation=0.01),
    ),
    "interaction_logistic_pair_only": (
        "Ablation: the same without resume-only features",
        "interaction_pair_only",
        pointwise_logistic,
    ),
}
# Reported for insight but never eligible for selection: these were added after the
# first test results had been seen.
ABLATIONS = {"interaction_logistic_pair_only"}
# Features that describe the resume alone. Almost every test resume also appears in the
# training split (paired with other jobs), so a model could use these to recognise a
# resume it has seen labelled before. The ablation checks how much that matters.
RESUME_ONLY_FEATURES = ["skills_resume_count", "skills_extra_count", "resume_log_words"]
METHOD_LABELS = FIXED_METHODS | {name: spec[0] for name, spec in LEARNED_METHODS.items()}
METHOD_LABELS["fine_tuned"] = "Fine-tuned bi-encoder"
METRIC_LABELS = {
    "auc_within_job": "AUC within each job",
    "auc_all_pairs": "AUC over all pairs",
    "spearman": "Spearman",
}
BASELINES = ["v1_whole_document", "tfidf"]
LABEL_NAMES = {0: "No Fit", 1: "Potential Fit", 2: "Good Fit"}


def score_pairs(pairs: pd.DataFrame, analyzer: ResumeAnalyzer, encoder, name: str) -> pd.DataFrame:
    """One row per pair: the v2 component scores, the v2 overall score and the v1 score."""
    rows = []
    for i, pair in enumerate(pairs.itertuples(), start=1):
        result = analyzer.analyze(pair.resume, pair.job)
        row = {c.key: c.score for c in result.components}
        row["v2_overall"] = result.match_score
        # v1 embedded each whole document as one string, which the model cuts at 256 tokens
        resume_vector, job_vector = encoder.encode([pair.resume, pair.job])
        row["v1_whole_document"] = float(resume_vector @ job_vector) * 100
        rows.append(row)
        if i % 1000 == 0:
            logger.info("%s: scored %d of %d pairs", name, i, len(pairs))
    return pd.DataFrame(rows, dtype=float)


def cached(path: Path, compute) -> pd.DataFrame:
    """Runs a slow step once and keeps its output; delete the file to recompute."""
    if path.exists():
        logger.info("Reading %s", path.name)
        return pd.read_csv(path)
    frame = compute()
    frame.to_csv(path, index=False)
    return frame


def tfidf_scores(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Lexical baseline: TF-IDF cosine similarity, with vocabulary learnt on the training split."""
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    vectorizer.fit(pd.concat([train["resume"], train["job"]]))
    resumes = vectorizer.transform(test["resume"])
    jobs = vectorizer.transform(test["job"])
    return np.asarray(resumes.multiply(jobs).sum(axis=1)).ravel() * 100


def resume_prior(train: pd.DataFrame, pairs: pd.DataFrame, same_split: bool) -> np.ndarray:
    """Leakage check: scores a pair by how its resume was labelled in training, ignoring the job.

    For training pairs the pair's own label is left out of its resume's average.
    """
    totals = train.groupby("resume")["label"].agg(["sum", "count"])
    total = pairs["resume"].map(totals["sum"]).fillna(0).to_numpy(dtype=float)
    count = pairs["resume"].map(totals["count"]).fillna(0).to_numpy(dtype=float)
    if same_split:
        total, count = total - pairs["label"].to_numpy(), count - 1
    overall = train["label"].mean()
    return np.where(count > 0, total / np.maximum(count, 1), overall)


def unseen_resume_scores(make_model, x, labels, jobs, folds) -> np.ndarray:
    """Test scores from models that saw neither the pair's job nor its resume.

    For each fold of resumes, a model is fitted on the training pairs whose resume is in
    another fold, and scores the test pairs whose resume is in this one.
    """
    scores = np.zeros(len(x["test"]))
    for fold in np.unique(folds["test"]):
        keep = folds["train"] != fold
        model = make_model().fit(x["train"][keep], labels["train"][keep], jobs["train"][keep])
        scores[folds["test"] == fold] = model.score(x["test"][folds["test"] == fold])
    return scores


def measure_strata(labels, jobs, strata, scores, baseline=None, resamples=1000) -> dict:
    """Within-stratum AUC (or its difference from a baseline) with a job-level interval."""
    value = auc_within_job(labels, scores, strata)
    if baseline is not None:
        value -= auc_within_job(labels, baseline, strata)
    low, high = job_bootstrap(labels, scores, jobs, baseline, resamples, strata=strata)
    return {"value": round(value, 4), "ci95": [round(low, 4), round(high, 4)]}

def component_features(scores: pd.DataFrame, fill: pd.Series) -> np.ndarray:
    """The four v2 components. One that could not be assessed gets its training mean."""
    return scores[COMPONENTS].fillna(fill).to_numpy()


def measure(labels, jobs, scores, baseline=None, resamples=1000) -> dict:
    """Every metric for one method (or its difference from a baseline), with 95% intervals."""

    def entry(value, baseline_value, interval):
        if baseline is not None:
            value -= baseline_value
        return {"value": round(value, 4), "ci95": [round(v, 4) for v in interval]}

    other = (lambda metric: metric(labels, baseline)) if baseline is not None else (lambda m: 0)
    return {
        "auc_within_job": entry(
            auc_within_job(labels, scores, jobs),
            auc_within_job(labels, baseline, jobs) if baseline is not None else 0,
            job_bootstrap(labels, scores, jobs, baseline, resamples),
        ),
        "auc_all_pairs": entry(
            auc_all_pairs(labels, scores),
            other(auc_all_pairs),
            pair_bootstrap(auc_all_pairs, labels, scores, baseline, resamples),
        ),
        "spearman": entry(
            spearman(labels, scores),
            other(spearman),
            pair_bootstrap(spearman, labels, scores, baseline, resamples),
        ),
    }


def render_markdown(report: dict) -> str:
    def cells(metrics: dict, signed: bool = False) -> str:
        spec = "+.3f" if signed else ".3f"
        return " | ".join(
            f"{m['value']:{spec}} ({m['ci95'][0]:{spec}} to {m['ci95'][1]:{spec}})"
            for m in metrics.values()
        )

    header = " | ".join(METRIC_LABELS.values())
    divider = " | ".join("---" for _ in METRIC_LABELS)
    counts = ", ".join(f"{count} {name}" for name, count in report["test_label_counts"].items())
    lines = [
        "# Evaluation results",
        "",
        f"Generated by `python -m evaluation.run` on `{report['dataset']}`.",
        "",
        f"- Test split: {report['test_pairs']} resume-job pairs ({counts}) across "
        f"{report['test_jobs']} jobs, {report['test_jobs_ranked']} of which have both Good Fit "
        "and No Fit candidates and so count towards the within-job AUC.",
        f"- Training split: {report['train_pairs']} pairs across {report['train_jobs']} jobs. "
        f"No test job appears in training; {report['shared_resumes']} of the "
        f"{report['test_resumes']} test resumes do.",
        f"- Embedding model: `{report['embedding_model']}`.",
        "",
        "AUC is the chance that a Good Fit pair outscores a No Fit pair: 0.5 is chance, 1.0 is "
        "perfect. Brackets are 95% bootstrap confidence intervals (jobs are resampled for the "
        "within-job AUC, pairs for the other two).",
        "",
        "## Test results",
        "",
        f"| Method | {header} |",
        f"| --- | {divider} |",
    ]
    for name, metrics in report["test"].items():
        lines.append(f"| {METHOD_LABELS[name]} | {cells(metrics)} |")

    lines += [
        "",
        "## Model selection (training split only)",
        "",
        "Within-job AUC from 5-fold cross-validation over jobs: each training pair is scored "
        "by a model that never saw its job. Methods that need no training are scored directly. "
        "The test split played no part in choosing a method.",
        "",
        "| Method | Cross-validated AUC within each job |",
        "| --- | --- |",
    ]
    for name, value in sorted(report["validation"].items(), key=lambda item: -item[1]):
        marker = " **(selected)**" if name == report["selected"] else ""
        lines.append(f"| {METHOD_LABELS[name]}{marker} | {value:.3f} |")

    lines += [
        "",
        "## Differences on the test split",
        "",
        "An interval that excludes zero means the difference is unlikely to be noise.",
        "",
        f"| Comparison | {header} |",
        f"| --- | {divider} |",
    ]
    for comparison in report["comparisons"]:
        label = f"{METHOD_LABELS[comparison['method']]} − {METHOD_LABELS[comparison['baseline']]}"
        lines.append(f"| {label} | {cells(comparison['metrics'], signed=True)} |")

    unseen = report["unseen"]
    lines += [
        "",
        "## Unseen-resume protocol",
        "",
        f"{report['shared_resumes']} of the {report['test_resumes']} test resumes also appear in "
        "the training split, paired with other jobs. The leakage check in the first table shows "
        "why that matters: ranking candidates purely by how their resume was labelled in "
        "training, without reading the job, already scores well. A trained model can therefore "
        "gain from remembering resumes, which would not carry over to new candidates.",
        "",
        f"Here resumes are divided into {unseen['folds']} folds, and each test pair is scored by "
        "a model trained without any resume from that pair's fold, so the model has seen neither "
        "the job nor the resume. Candidates are compared only with others in the same job and "
        f"fold ({unseen['strata_ranked']} such groups have both Good Fit and No Fit candidates), "
        "so these numbers are not directly comparable with the first table. Methods that need "
        "no training are unaffected by the protocol and are re-measured on the same groups.",
        "",
        "| Method | AUC within each job and fold |",
        "| --- | --- |",
    ]
    for name, m in unseen["methods"].items():
        interval = f"{m['ci95'][0]:.3f} to {m['ci95'][1]:.3f}"
        lines.append(f"| {METHOD_LABELS[name]} | {m['value']:.3f} ({interval}) |")
    lines += ["", "| Comparison | Difference |", "| --- | --- |"]
    for comparison in unseen["comparisons"]:
        m = comparison["metric"]
        label = f"{METHOD_LABELS[comparison['method']]} − {METHOD_LABELS[comparison['baseline']]}"
        lines.append(
            f"| {label} | {m['value']:+.3f} ({m['ci95'][0]:+.3f} to {m['ci95'][1]:+.3f}) |"
        )

    coverage = ", ".join(f"{k} {v}" for k, v in report["component_coverage"].items())
    lines += [
        "",
        "## Details",
        "",
        f"- Share of test pairs where each v2 component could be assessed: {coverage}.",
        "- Largest weights of the pairwise ranker on interaction features (standardised "
        "inputs): "
        + ", ".join(f"`{k}` {v:+.2f}" for k, v in report["ranker_weights"].items())
        + ".",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-dir", type=Path, default=HERE / ".cache")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--limit", type=int, help="Use a random N pairs of each split")
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cache = args.cache_dir

    train, test = load_split("train", cache), load_split("test", cache)
    if args.limit:
        train = train.sample(min(args.limit, len(train)), random_state=0).reset_index(drop=True)
        test = test.sample(min(args.limit, len(test)), random_state=0).reset_index(drop=True)
    labels = {"train": train["label"].to_numpy(), "test": test["label"].to_numpy()}
    jobs = {"train": pd.factorize(train["job"])[0], "test": pd.factorize(test["job"])[0]}

    settings = Settings()
    encoder = CachingEncoder(
        SentenceTransformerEncoder(settings.embedding_model), cache / "embeddings.npz"
    )
    analyzer = ResumeAnalyzer(encoder, settings=settings)
    index = DocumentIndex(analyzer, encoder)
    index.add(
        pd.unique(pd.concat([train["resume"], test["resume"]])).tolist(),
        pd.unique(pd.concat([train["job"], test["job"]])).tolist(),
    )

    splits = {"train": train, "test": test}
    base = {
        name: cached(
            cache / f"scores_{name}_{len(pairs)}.csv",
            lambda pairs=pairs, name=name: score_pairs(pairs, analyzer, encoder, name),
        )
        for name, pairs in splits.items()
    }
    interaction = {
        name: cached(
            cache / f"features_{name}_{len(pairs)}.csv",
            lambda pairs=pairs: pair_features(pairs, index),
        )
        for name, pairs in splits.items()
    }
    component_means = base["train"][COMPONENTS].mean().fillna(50)
    features = {
        "components": {n: component_features(base[n], component_means) for n in splits},
        "interaction": {n: interaction[n].to_numpy() for n in splits},
        "interaction_pair_only": {
            n: interaction[n].drop(columns=RESUME_ONLY_FEATURES).to_numpy() for n in splits
        },
        "embedding": {n: embedding_features(pairs, index) for n, pairs in splits.items()},
    }

    # Scores of every method on both splits. Training-split scores of learned methods are
    # cross-validated, so that they can be compared fairly with the fixed methods.
    scores = {"train": {}, "test": {}}
    scores["train"]["tfidf"] = tfidf_scores(train, train)
    scores["test"]["tfidf"] = tfidf_scores(train, test)
    for name in FIXED_METHODS:
        if name not in ("tfidf", "resume_prior"):
            for split in splits:
                # A component that could not be assessed gets the neutral mid-point
                scores[split][name] = base[split][name].fillna(50).to_numpy()

    scores["train"]["resume_prior"] = resume_prior(train, train, same_split=True)
    scores["test"]["resume_prior"] = resume_prior(train, test, same_split=False)

    ranker_weights = {}
    for name, (_, feature_set, make_model) in LEARNED_METHODS.items():
        logger.info("Fitting %s", name)
        x = features[feature_set]
        scores["train"][name] = cross_validated_scores(
            make_model, x["train"], labels["train"], jobs["train"]
        )
        model = make_model().fit(x["train"], labels["train"], jobs["train"])
        scores["test"][name] = model.score(x["test"])
        if name == "interaction_ranker":
            order = np.argsort(-np.abs(model.weights))[:6]
            ranker_weights = {
                interaction["train"].columns[i]: round(float(model.weights[i]), 3) for i in order
            }

    fine_tuned = cache / FINE_TUNED_SCORES
    if fine_tuned.exists() and not args.limit:
        scores["test"]["fine_tuned"] = pd.read_csv(fine_tuned)["score"].to_numpy()

    validation = {
        name: round(auc_within_job(labels["train"], s, jobs["train"]), 4)
        for name, s in scores["train"].items()
    }
    selected = max((m for m in LEARNED_METHODS if m not in ABLATIONS), key=validation.get)
    contenders = [selected, *ABLATIONS]
    if "fine_tuned" in scores["test"]:
        contenders.append("fine_tuned")

    y, groups = labels["test"], jobs["test"]
    report = {
        "dataset": DATASET,
        "embedding_model": settings.embedding_model,
        "train_pairs": len(train),
        "train_jobs": int(train["job"].nunique()),
        "test_pairs": len(test),
        "test_jobs": int(test["job"].nunique()),
        "test_jobs_ranked": len(per_job_auc(y, scores["test"]["tfidf"], groups)),
        "test_resumes": int(test["resume"].nunique()),
        "test_label_counts": {
            LABEL_NAMES[k]: int(v) for k, v in test["label"].value_counts().sort_index().items()
        },
        "shared_resumes": int(test["resume"].drop_duplicates().isin(train["resume"]).sum()),
        "shared_jobs": int(test["job"].drop_duplicates().isin(train["job"]).sum()),
        "component_coverage": {c: f"{base['test'][c].notna().mean():.0%}" for c in COMPONENTS},
        "ranker_weights": ranker_weights,
        "validation": validation,
        "selected": selected,
        "test": {
            name: measure(y, groups, s, resamples=args.resamples)
            for name, s in scores["test"].items()
        },
        "comparisons": [
            {
                "method": method,
                "baseline": baseline,
                "metrics": measure(
                    y, groups, scores["test"][method], scores["test"][baseline], args.resamples
                ),
            }
            for method in ["v2_overall", *contenders]
            for baseline in BASELINES
        ],
    }

    # --- Unseen-resume protocol ---
    # Nearly every test resume also appears in training, paired with other jobs, so a
    # trained model can score well just by remembering how a resume was labelled. Here
    # each test pair is scored by a model trained without that pair's resume.
    folds = {
        name: pairs["resume"].map(lambda r: resume_fold(r, UNSEEN_FOLDS)).to_numpy()
        for name, pairs in splits.items()
    }
    # Candidates are only compared with others scored by the same model
    strata = groups * UNSEEN_FOLDS + folds["test"]
    unseen = {name: scores["test"][name] for name in ["tfidf", "v1_whole_document", "v2_overall"]}
    for name in [selected, *ABLATIONS]:
        make_model = LEARNED_METHODS[name][2]
        unseen[name] = unseen_resume_scores(
            make_model, features[LEARNED_METHODS[name][1]], labels, jobs, folds
        )
    unseen_fine_tuned = cache / UNSEEN_SCORES
    if unseen_fine_tuned.exists() and not args.limit:
        stored = pd.read_csv(unseen_fine_tuned)
        if stored["done"].all():
            unseen["fine_tuned"] = stored["score"].to_numpy()
    report["unseen"] = {
        "folds": UNSEEN_FOLDS,
        "strata_ranked": len(per_job_auc(y, unseen["tfidf"], strata)),
        "methods": {
            name: measure_strata(y, groups, strata, s, resamples=args.resamples)
            for name, s in unseen.items()
        },
        "comparisons": [
            {
                "method": method,
                "baseline": baseline,
                "metric": measure_strata(
                    y, groups, strata, unseen[method], unseen[baseline], args.resamples
                ),
            }
            for method in unseen
            if method not in BASELINES
            for baseline in BASELINES
        ],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "RESULTS.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


if __name__ == "__main__":
    main()
