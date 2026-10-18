"""Learned scorers. Each is fitted on labelled pairs and then scores unseen pairs."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GOOD_FIT, NO_FIT = 2, 0
MAX_PAIRS_PER_JOB = 300


class PointwiseClassifier:
    """Predicts Good Fit vs No Fit for each pair on its own; the probability is the score."""

    def __init__(self, classifier):
        self._model = classifier

    def fit(self, features: np.ndarray, labels: np.ndarray, jobs: np.ndarray):
        # Potential Fit is left out, so the model learns the clearest contrast
        clear = labels != 1
        self._model.fit(features[clear], labels[clear] == GOOD_FIT)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(features)[:, 1]


class PairwiseRanker:
    """Learns to order two candidates for the same job (the RankSVM idea, with a logistic loss).

    Training examples are differences between the features of a Good Fit and a No Fit
    resume for one job. Anything that is constant for a job cancels out, so the model
    cannot lean on some jobs simply having more good candidates than others.
    """

    def __init__(self, regularisation: float = 1.0, seed: int = 0):
        self._scaler = StandardScaler()
        self._model = LogisticRegression(C=regularisation, fit_intercept=False, max_iter=2000)
        self._seed = seed

    def fit(self, features: np.ndarray, labels: np.ndarray, jobs: np.ndarray):
        scaled = self._scaler.fit_transform(features)
        rng = np.random.default_rng(self._seed)
        differences = []
        for job in np.unique(jobs):
            good = np.flatnonzero((jobs == job) & (labels == GOOD_FIT))
            poor = np.flatnonzero((jobs == job) & (labels == NO_FIT))
            if not len(good) or not len(poor):
                continue
            pairs = np.array([(g, p) for g in good for p in poor])
            if len(pairs) > MAX_PAIRS_PER_JOB:  # keep jobs with many candidates from dominating
                pairs = pairs[rng.choice(len(pairs), MAX_PAIRS_PER_JOB, replace=False)]
            differences.append(scaled[pairs[:, 0]] - scaled[pairs[:, 1]])

        better = np.concatenate(differences)
        # Each difference is used both ways round, so the classes are balanced
        self._model.fit(
            np.concatenate([better, -better]), np.repeat([1, 0], len(better))
        )
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        return self._scaler.transform(features) @ self._model.coef_[0]

    @property
    def weights(self) -> np.ndarray:
        return self._model.coef_[0]


def pointwise_logistic() -> PointwiseClassifier:
    return PointwiseClassifier(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))


def pointwise_boosting() -> PointwiseClassifier:
    return PointwiseClassifier(
        HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=0
        )
    )


def cross_validated_scores(make_model, features, labels, jobs, folds: int = 5) -> np.ndarray:
    """Scores for every training pair from a model that never saw that pair's job."""
    scores = np.zeros(len(labels))
    for fit_rows, held_out in GroupKFold(folds).split(features, labels, groups=jobs):
        model = make_model().fit(features[fit_rows], labels[fit_rows], jobs[fit_rows])
        scores[held_out] = model.score(features[held_out])
    return scores
