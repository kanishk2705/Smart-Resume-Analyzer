"""Ranking metrics and bootstrap confidence intervals."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

GOOD_FIT, NO_FIT = 2, 0


def auc_all_pairs(labels: np.ndarray, scores: np.ndarray) -> float:
    """Chance that a random Good Fit pair outscores a random No Fit pair. 0.5 is chance."""
    keep = (labels == GOOD_FIT) | (labels == NO_FIT)
    return float(roc_auc_score(labels[keep] == GOOD_FIT, scores[keep]))


def spearman(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank correlation between the score and the three-level label."""
    return float(spearmanr(labels, scores).statistic)


def per_job_auc(labels: np.ndarray, scores: np.ndarray, jobs: np.ndarray) -> np.ndarray:
    """Good-vs-No-Fit AUC for each job that has both kinds of candidate."""
    aucs = []
    for job in np.unique(jobs):
        mine = (jobs == job) & ((labels == GOOD_FIT) | (labels == NO_FIT))
        is_good = labels[mine] == GOOD_FIT
        if is_good.any() and not is_good.all():
            aucs.append(roc_auc_score(is_good, scores[mine]))
    return np.array(aucs)


def auc_within_job(labels: np.ndarray, scores: np.ndarray, jobs: np.ndarray) -> float:
    """Mean of the per-job AUCs.

    This is the recruiter's question: for one job, are the good candidates ranked above
    the poor ones? It is unaffected by some jobs scoring every candidate higher than others.
    """
    aucs = per_job_auc(labels, scores, jobs)
    return float(aucs.mean()) if len(aucs) else float("nan")


def _percentile_interval(values: list[float]) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def pair_bootstrap(
    metric, labels, scores, baseline=None, resamples: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% interval for a metric over all pairs, resampling pairs with replacement.

    With `baseline`, the interval is for metric(scores) - metric(baseline) on the same
    resampled pairs, which is what a comparison between two methods needs.
    """
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        s = rng.integers(0, len(labels), len(labels))
        value = metric(labels[s], scores[s])
        if baseline is not None:
            value -= metric(labels[s], baseline[s])
        values.append(value)
    return _percentile_interval(values)


def job_bootstrap(
    labels, scores, jobs, baseline=None, resamples: int = 1000, seed: int = 0, strata=None
) -> tuple[float, float]:
    """95% interval for the within-job AUC, resampling whole jobs with replacement.

    Pairs for the same job are not independent, so the job is the unit that is resampled.
    With `strata`, the AUC is computed within each stratum (a subdivision of a job) and
    all of a job's strata are resampled together.
    """
    groups = jobs if strata is None else strata
    qualifying = _qualifying_groups(labels, groups)
    aucs = per_job_auc(labels, scores, groups)
    if baseline is not None:
        aucs = aucs - per_job_auc(labels, baseline, groups)
    # The job each qualifying group belongs to
    owner = np.array([jobs[groups == group][0] for group in qualifying])
    owners = np.unique(owner)
    members = [np.flatnonzero(owner == job) for job in owners]

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        chosen = np.concatenate([members[i] for i in rng.integers(0, len(owners), len(owners))])
        values.append(aucs[chosen].mean())
    return _percentile_interval(values)


def _qualifying_groups(labels: np.ndarray, groups: np.ndarray) -> list:
    """Groups with both a Good Fit and a No Fit pair, in the order per_job_auc reports them."""
    qualifying = []
    for group in np.unique(groups):
        mine = labels[groups == group]
        if (mine == GOOD_FIT).any() and (mine == NO_FIT).any():
            qualifying.append(group)
    return qualifying
