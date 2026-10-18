import numpy as np
import pandas as pd
import pytest

from core.scoring import ResumeAnalyzer
from evaluation.data import restore_line_breaks, resume_fold
from evaluation.features import (
    CachingEncoder,
    DocumentIndex,
    embedding_features,
    interaction_features,
)
from evaluation.metrics import (
    auc_all_pairs,
    auc_within_job,
    job_bootstrap,
    pair_bootstrap,
    per_job_auc,
    spearman,
)
from evaluation.models import PairwiseRanker, cross_validated_scores, pointwise_logistic
from evaluation.run import (
    measure,
    measure_strata,
    resume_prior,
    tfidf_scores,
    unseen_resume_scores,
)
from tests.conftest import FakeEncoder

NO, POTENTIAL, GOOD = 0, 1, 2


def arrays(*columns):
    return [np.array(column) for column in columns]


# --- Metrics ---


def test_auc_is_one_when_every_good_fit_outscores_every_no_fit():
    labels, scores = arrays([GOOD, GOOD, NO, NO], [90, 80, 20, 10])

    assert auc_all_pairs(labels, scores) == 1.0
    assert auc_all_pairs(labels, -scores) == 0.0


def test_auc_ignores_potential_fit_pairs():
    labels, scores = arrays([GOOD, NO, POTENTIAL], [90, 10, -100])

    assert auc_all_pairs(labels, scores) == 1.0


def test_within_job_auc_ignores_score_differences_between_jobs():
    # Job 0 scores everyone high and job 1 scores everyone low, but each ranks correctly
    labels, scores, jobs = arrays([GOOD, NO, GOOD, NO], [90, 80, 20, 10], [0, 0, 1, 1])

    assert auc_within_job(labels, scores, jobs) == 1.0
    assert auc_all_pairs(labels, scores) == 0.75


def test_per_job_auc_skips_jobs_without_both_kinds_of_candidate():
    labels, scores, jobs = arrays([GOOD, NO, NO, NO, GOOD], [90, 10, 50, 60, 5], [0, 0, 1, 1, 2])

    assert per_job_auc(labels, scores, jobs).tolist() == [1.0]


def test_within_job_auc_is_nan_when_no_job_qualifies():
    labels, scores, jobs = arrays([GOOD, NO], [90, 10], [0, 1])

    assert np.isnan(auc_within_job(labels, scores, jobs))


def test_spearman_uses_all_three_label_levels():
    labels, scores = arrays([NO, POTENTIAL, GOOD], [10, 50, 90])

    assert spearman(labels, scores) == pytest.approx(1.0)
    assert spearman(labels, -scores) == pytest.approx(-1.0)


def noisy_benchmark(pairs=300, job_count=15, seed=1):
    rng = np.random.default_rng(seed)
    labels = rng.choice([NO, GOOD], pairs)
    jobs = rng.integers(0, job_count, pairs)
    scores = labels * 10 + rng.normal(0, 12, pairs)
    return labels, jobs, scores


def test_bootstrap_intervals_bracket_the_estimate_and_are_reproducible():
    labels, jobs, scores = noisy_benchmark()

    low, high = pair_bootstrap(auc_all_pairs, labels, scores, resamples=200)
    assert low < auc_all_pairs(labels, scores) < high
    assert (low, high) == pair_bootstrap(auc_all_pairs, labels, scores, resamples=200)

    low, high = job_bootstrap(labels, scores, jobs, resamples=200)
    assert low < auc_within_job(labels, scores, jobs) < high


def test_bootstrap_difference_of_a_method_with_itself_is_zero():
    labels, jobs, scores = noisy_benchmark()

    assert pair_bootstrap(auc_all_pairs, labels, scores, scores, resamples=50) == (0.0, 0.0)
    assert job_bootstrap(labels, scores, jobs, scores, resamples=50) == (0.0, 0.0)


def test_measure_reports_differences_from_a_baseline():
    labels, jobs, scores = noisy_benchmark()

    alone = measure(labels, jobs, scores, resamples=20)
    versus_reversed = measure(labels, jobs, scores, -scores, resamples=20)

    assert set(alone) == {"auc_within_job", "auc_all_pairs", "spearman"}
    auc = alone["auc_all_pairs"]["value"]
    assert versus_reversed["auc_all_pairs"]["value"] == pytest.approx(auc - (1 - auc), abs=1e-3)
    assert versus_reversed["auc_within_job"]["ci95"][0] > 0


# --- Data ---


def test_restore_line_breaks_after_glued_sentences_and_headings():
    text = "Summary7+ years in BI projects.Proven ability to deliver.\nExperienceBI Developer"

    assert restore_line_breaks(text) == (
        "Summary\n7+ years in BI projects.\nProven ability to deliver.\nExperience\nBI Developer"
    )


def test_restore_line_breaks_leaves_normal_text_alone():
    text = "Built 3.5x faster pipelines. Used Node.js and U.S. data.\nExperienced engineer"

    assert restore_line_breaks(text) == text


# --- Features ---


class CountingEncoder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array([[len(text)] for text in texts], dtype=np.float32)


def test_caching_encoder_embeds_each_text_once():
    inner = CountingEncoder()
    encoder = CachingEncoder(inner)

    first = encoder.encode(["aa", "b", "aa"])
    second = encoder.encode(["b", "cccc"])

    assert first.tolist() == [[2.0], [1.0], [2.0]]
    assert second.tolist() == [[1.0], [4.0]]
    assert inner.calls == [["aa", "b"], ["cccc"]]


def test_caching_encoder_persists_to_disk(tmp_path):
    path = tmp_path / "cache" / "embeddings.npz"
    first = CachingEncoder(CountingEncoder(), path)
    first.encode(["aa", "b"])
    first.save()

    inner = CountingEncoder()
    reloaded = CachingEncoder(inner, path)

    assert reloaded.encode(["b", "aa"]).tolist() == [[1.0], [2.0]]
    assert inner.calls == []


@pytest.fixture
def index():
    encoder = CachingEncoder(FakeEncoder())
    index = DocumentIndex(ResumeAnalyzer(encoder), encoder)
    index.add(
        resumes=[
            "Experience\n• I write Python and SQL services every day.\n• I deploy with Docker.",
            "Experience\n• I ran marketing campaigns for retail brands.",
        ],
        jobs=["We need strong Python and SQL skills.\nYou will deploy with Docker containers."],
    )
    return index


RELEVANT = "Experience\n• I write Python and SQL services every day.\n• I deploy with Docker."
IRRELEVANT = "Experience\n• I ran marketing campaigns for retail brands."
JOB = "We need strong Python and SQL skills.\nYou will deploy with Docker containers."


def test_interaction_features_separate_relevant_from_irrelevant_resumes(index):
    relevant = interaction_features(RELEVANT, JOB, index)
    irrelevant = interaction_features(IRRELEVANT, JOB, index)

    assert relevant["coverage_mean"] == pytest.approx(1.0)
    assert irrelevant["coverage_mean"] == 0.0
    assert relevant["skills_matched_share"] == 1.0
    assert relevant["skills_matched_count"] == 3
    assert irrelevant["skills_matched_count"] == 0
    assert relevant["pooled_cosine"] > irrelevant["pooled_cosine"]
    assert all(np.isfinite(value) for value in relevant.values())


def test_embedding_features_have_one_row_per_pair(index):
    pairs = pd.DataFrame({"resume": [RELEVANT, IRRELEVANT], "job": [JOB, JOB]})

    features = embedding_features(pairs, index)

    assert features.shape == (2, 2 * len(FakeEncoder().encode(["x"])[0]))
    assert np.isfinite(features).all()


def test_tfidf_scores_reward_shared_vocabulary():
    train = pd.DataFrame({"resume": ["python sql docker"], "job": ["marketing sales retail"]})
    test = pd.DataFrame({
        "resume": ["python sql developer", "python sql developer"],
        "job": ["python sql engineer", "marketing sales manager"],
    })

    matching, unrelated = tfidf_scores(train, test)

    assert matching > unrelated == 0


# --- Models ---


def synthetic_ranking_problem(seed=0):
    """Feature 0 decides who fits within a job. Feature 1 is a per-job offset that is
    larger for jobs with many good candidates: informative across jobs, useless within one."""
    rng = np.random.default_rng(seed)
    jobs = np.repeat(np.arange(30), 20)
    good_rate = rng.uniform(0.2, 0.8, 30)[jobs]
    labels = np.where(rng.uniform(size=600) < good_rate, GOOD, NO)
    signal = (labels == GOOD) * 1.0 + rng.normal(0, 0.7, 600)
    features = np.column_stack([signal, good_rate * 5 + rng.normal(0, 0.1, 600)])
    return features, labels, jobs


def test_pairwise_ranker_learns_the_within_job_signal():
    features, labels, jobs = synthetic_ranking_problem()

    ranker = PairwiseRanker().fit(features, labels, jobs)

    assert ranker.weights[0] > abs(ranker.weights[1])
    assert auc_within_job(labels, ranker.score(features), jobs) > 0.8


def test_pairwise_ranker_is_unaffected_by_a_constant_added_to_one_jobs_features():
    features, labels, jobs = synthetic_ranking_problem()
    shifted = features.copy()
    shifted[jobs == 0] += 1000

    ranker = PairwiseRanker().fit(shifted, labels, jobs)

    # Differences within a job cancel the offset, so the ranking signal is still found
    assert ranker.weights[0] > 0
    assert auc_within_job(labels, ranker.score(shifted), jobs) > 0.8


def test_pairwise_ranker_skips_jobs_with_one_kind_of_candidate():
    features, labels, jobs = synthetic_ranking_problem()
    labels[jobs < 5] = NO

    ranker = PairwiseRanker().fit(features, labels, jobs)

    assert np.isfinite(ranker.score(features)).all()


def test_pointwise_model_ignores_potential_fit_when_training():
    features, labels, jobs = synthetic_ranking_problem()
    labels[:50] = POTENTIAL
    features[:50, 0] = -100  # would wreck the model if these rows were used as positives

    model = pointwise_logistic().fit(features, labels, jobs)

    assert auc_within_job(labels[50:], model.score(features[50:]), jobs[50:]) > 0.8


def test_cross_validated_scores_come_from_models_that_never_saw_the_job():
    features, labels, jobs = synthetic_ranking_problem()
    seen = []

    class Recorder(PairwiseRanker):
        def fit(self, x, y, groups):
            seen.append(set(groups))
            return super().fit(x, y, groups)

    scores = cross_validated_scores(Recorder, features, labels, jobs, folds=5)

    assert len(seen) == 5
    assert all(len(fold_jobs) == 24 for fold_jobs in seen)
    assert set.intersection(*seen) == set()  # every job is held out exactly once
    assert auc_within_job(labels, scores, jobs) > 0.8


# --- Leakage check and unseen-resume protocol ---


def test_resume_fold_is_stable_and_spreads_resumes():
    folds = [resume_fold(f"resume number {i}", 4) for i in range(200)]

    assert folds == [resume_fold(f"resume number {i}", 4) for i in range(200)]
    assert set(folds) == {0, 1, 2, 3}


def test_resume_prior_is_the_resumes_average_training_label():
    train = pd.DataFrame({
        "resume": ["a", "a", "b"], "job": ["j1", "j2", "j1"], "label": [GOOD, GOOD, NO],
    })
    test = pd.DataFrame({"resume": ["a", "b", "new"], "job": ["j3"] * 3, "label": [NO] * 3})

    scores = resume_prior(train, test, same_split=False)

    # A resume never seen in training gets the overall average
    assert scores.tolist() == [2.0, 0.0, pytest.approx(4 / 3)]


def test_resume_prior_leaves_out_a_training_pairs_own_label():
    train = pd.DataFrame({
        "resume": ["a", "a", "b"], "job": ["j1", "j2", "j1"], "label": [GOOD, NO, GOOD],
    })

    scores = resume_prior(train, train, same_split=True)

    assert scores.tolist() == [0.0, 2.0, pytest.approx(4 / 3)]


def test_unseen_resume_scores_never_train_on_the_scored_resumes_fold():
    features, labels, jobs = synthetic_ranking_problem()
    folds = np.arange(600) % 4
    trained_on = []

    class Recorder(PairwiseRanker):
        def fit(self, x, y, groups):
            trained_on.append(len(x))
            return super().fit(x, y, groups)

    scores = unseen_resume_scores(
        Recorder,
        {"train": features, "test": features},
        {"train": labels},
        {"train": jobs},
        {"train": folds, "test": folds},
    )

    assert trained_on == [450] * 4
    assert auc_within_job(labels, scores, jobs * 4 + folds) > 0.75


def test_measure_strata_compares_only_within_a_stratum():
    # Stratum 1 of the job is scored on a different scale; each stratum ranks correctly
    labels, scores, jobs, strata = arrays(
        [GOOD, NO, GOOD, NO], [9, 8, 2, 1], [0, 0, 0, 0], [0, 0, 1, 1]
    )

    assert measure_strata(labels, jobs, strata, scores, resamples=20)["value"] == 1.0
    assert auc_within_job(labels, scores, jobs) == 0.75


def test_strata_of_one_job_are_resampled_together():
    labels, jobs, scores = noisy_benchmark(pairs=600, job_count=6)
    strata = jobs * 4 + np.arange(600) % 4

    by_job = job_bootstrap(labels, scores, jobs, resamples=300, strata=strata)
    # Treating every stratum as independent gives a narrower, over-confident interval
    independent = job_bootstrap(labels, scores, strata, resamples=300)

    assert by_job[0] < auc_within_job(labels, scores, strata) < by_job[1]
    assert (by_job[1] - by_job[0]) > 0.8 * (independent[1] - independent[0])