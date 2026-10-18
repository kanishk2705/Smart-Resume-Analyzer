"""Fine-tunes the embedding model on the training split, then scores the test split.

    python -m evaluation.finetune                  # standard protocol
    python -m evaluation.finetune --unseen-resumes # no test resume is seen in training

The base model ranks candidates by the cosine similarity of resume and job embeddings.
Fine-tuning adjusts the model so that, for the same job, a Good Fit resume comes out
closer to the job than a No Fit resume. Test scores are written into the cache directory,
where `python -m evaluation.run` picks them up.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupShuffleSplit

from core.config import Settings
from evaluation.data import load_split, resume_fold
from evaluation.metrics import auc_within_job

logger = logging.getLogger("evaluation")

HERE = Path(__file__).parent
FINE_TUNED_SCORES = "scores_finetuned_test.csv"
UNSEEN_SCORES = "scores_finetuned_unseen_test.csv"
UNSEEN_FOLDS = 4
GOOD_FIT, NO_FIT = 2, 0
MARGIN = 0.2


def triplets(pairs: pd.DataFrame, per_job: int, rng) -> list[tuple[str, str, str]]:
    """(job, Good Fit resume, No Fit resume) examples, a capped number from each job."""
    examples = []
    for job, group in pairs.groupby("job"):
        good = group.loc[group["label"] == GOOD_FIT, "resume"].tolist()
        poor = group.loc[group["label"] == NO_FIT, "resume"].tolist()
        if not good or not poor:
            continue
        combinations = [(job, g, p) for g in good for p in poor]
        chosen = rng.choice(len(combinations), min(per_job, len(combinations)), replace=False)
        examples.extend(combinations[i] for i in chosen)
    return examples


def embed(model: SentenceTransformer, texts: list[str]) -> torch.Tensor:
    output = model(model.tokenize(texts))["sentence_embedding"]
    return torch.nn.functional.normalize(output, dim=1)


def score(model: SentenceTransformer, pairs: pd.DataFrame) -> np.ndarray:
    """Cosine similarity of each pair, embedding every unique document once."""
    model.eval()
    documents = pd.unique(pd.concat([pairs["resume"], pairs["job"]])).tolist()
    vectors = model.encode(
        documents, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    lookup = dict(zip(documents, vectors, strict=True))
    return np.array([lookup[p.resume] @ lookup[p.job] for p in pairs.itertuples()])


def within_job_auc(model, pairs: pd.DataFrame) -> float:
    jobs = pd.factorize(pairs["job"])[0]
    return auc_within_job(pairs["label"].to_numpy(), score(model, pairs), jobs)


def train_epoch(model, optimizer, pairs: pd.DataFrame, args, rng, name: str) -> None:
    examples = triplets(pairs, args.triplets_per_job, rng)
    order = rng.permutation(len(examples))
    steps = -(-len(order) // args.batch_size)
    model.train()
    started, running = time.time(), 0.0
    for step, start in enumerate(range(0, len(order), args.batch_size), start=1):
        batch = [examples[i] for i in order[start : start + args.batch_size]]
        job, good, poor = (embed(model, list(texts)) for texts in zip(*batch, strict=True))
        # The Good Fit resume should be closer to the job than the No Fit one, by a margin
        loss = torch.relu(MARGIN - (job * good).sum(1) + (job * poor).sum(1)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item()
        if step % 25 == 0:
            logger.info(
                "%s step %d/%d loss %.4f (%.0fs)",
                name, step, steps, running / 25, time.time() - started,
            )
            running = 0.0


def new_model(args):
    model = SentenceTransformer(Settings().embedding_model)
    return model, torch.optim.AdamW(model.parameters(), lr=args.learning_rate)


def standard_protocol(train, test, args, rng) -> None:
    """Trains on the training split, stopping at the epoch that does best on held-back jobs."""
    fit_rows, validation_rows = next(
        GroupShuffleSplit(1, test_size=0.2, random_state=0).split(train, groups=train["job"])
    )
    fit, validation = train.iloc[fit_rows], train.iloc[validation_rows]

    model, optimizer = new_model(args)
    history = [{"epoch": 0, "validation_auc": within_job_auc(model, validation)}]
    logger.info("Before fine-tuning: validation within-job AUC %.3f", history[0]["validation_auc"])

    best_auc, best_state = history[0]["validation_auc"], None
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, optimizer, fit, args, rng, f"epoch {epoch}")
        auc = within_job_auc(model, validation)
        history.append({"epoch": epoch, "validation_auc": auc})
        logger.info("After epoch %d: validation within-job AUC %.3f", epoch, auc)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Keep the epoch that did best on the held-back jobs; possibly the untouched model
    if best_state is None:
        logger.info("Fine-tuning never beat the base model on validation; scoring the base model")
        model, _ = new_model(args)
    else:
        model.load_state_dict(best_state)

    pd.DataFrame({"score": score(model, test)}).to_csv(
        args.cache_dir / FINE_TUNED_SCORES, index=False
    )
    pd.DataFrame(history).to_csv(args.cache_dir / "finetune_history.csv", index=False)
    logger.info("Test within-job AUC: %.3f", within_job_auc(model, test))


def unseen_resume_protocol(train, test, args, rng) -> None:
    """Scores every test pair with a model that saw neither its job nor its resume.

    Resumes are divided into folds. For each fold, a fresh model is trained for one epoch
    on the training pairs whose resume is in another fold, and scores the test pairs whose
    resume is in this one. One epoch is fixed in advance, so no validation data is needed.
    """
    train_folds = train["resume"].map(lambda r: resume_fold(r, UNSEEN_FOLDS)).to_numpy()
    test_folds = test["resume"].map(lambda r: resume_fold(r, UNSEEN_FOLDS)).to_numpy()
    scores = np.zeros(len(test))
    for fold in range(UNSEEN_FOLDS):
        model, optimizer = new_model(args)
        train_epoch(model, optimizer, train[train_folds != fold], args, rng, f"fold {fold}")
        scores[test_folds == fold] = score(model, test[test_folds == fold])
        logger.info("Fold %d done: scored %d test pairs", fold, (test_folds == fold).sum())
        # Written after every fold, so an interrupted run keeps what it finished
        pd.DataFrame({"score": scores, "fold": test_folds, "done": test_folds <= fold}).to_csv(
            args.cache_dir / UNSEEN_SCORES, index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-dir", type=Path, default=HERE / ".cache")
    parser.add_argument("--unseen-resumes", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--triplets-per-job", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    train, test = load_split("train", args.cache_dir), load_split("test", args.cache_dir)
    protocol = unseen_resume_protocol if args.unseen_resumes else standard_protocol
    protocol(train, test, args, rng)


if __name__ == "__main__":
    main()
