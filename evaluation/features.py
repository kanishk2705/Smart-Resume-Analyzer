"""Turns resume-job pairs into numeric features for the learned scorers."""

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from core.scoring import ResumeAnalyzer
from core.sections import split_sections
from core.skills import get_taxonomy

logger = logging.getLogger("evaluation")

TOP_K = 5
STRONG, PARTIAL = 0.50, 0.35


class CachingEncoder:
    """Remembers every embedding, since the benchmark reuses each document in many pairs.

    With a `path`, the cache is kept on disk between runs; embedding is the slow step.
    """

    def __init__(self, encoder, path: Path | None = None):
        self._encoder = encoder
        self._path = path
        self._cache: dict[str, np.ndarray] = {}  # keyed by a hash of the text
        if path and path.exists():
            stored = np.load(path, allow_pickle=False)
            self._cache = dict(zip(stored["keys"].tolist(), stored["vectors"], strict=True))
            logger.info("Loaded %d cached embeddings", len(self._cache))

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def encode(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(text) for text in texts]
        missing = {
            key: text for key, text in zip(keys, texts, strict=True) if key not in self._cache
        }
        if missing:
            vectors = self._encoder.encode(list(missing.values()))
            self._cache.update(zip(missing, vectors, strict=True))
        return np.array([self._cache[key] for key in keys])

    def save(self) -> None:
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                self._path,
                keys=np.array(list(self._cache)),
                vectors=np.array(list(self._cache.values()), dtype=np.float32),
            )

class DocumentIndex:
    """Each unique document split into units, embedded once, with its skills extracted."""

    def __init__(self, analyzer: ResumeAnalyzer, encoder: CachingEncoder):
        self._analyzer = analyzer
        self._encoder = encoder
        self._taxonomy = get_taxonomy()
        self._units: dict[str, list[str]] = {}
        self._skills: dict[str, set[str]] = {}

    def add(self, resumes: list[str], jobs: list[str]) -> None:
        for text in resumes:
            if text not in self._units:
                self._units[text] = self._analyzer.resume_units(text, split_sections(text))[0]
        for text in jobs:
            if text not in self._units:
                self._units[text] = self._analyzer.jd_units(text)
        for text in [*resumes, *jobs]:
            if text not in self._skills:
                self._skills[text] = {skill.name for skill in self._taxonomy.extract(text)}

        # One large call lets the model batch sentences of similar length together
        everything = [unit for units in self._units.values() for unit in units]
        logger.info("Embedding %d units and %d whole documents", len(everything), len(self._units))
        self._encoder.encode(everything + list(self._units))
        self._encoder.save()

    def units(self, text: str) -> np.ndarray:
        return self._encoder.encode(self._units[text])

    def whole(self, text: str) -> np.ndarray:
        """The document embedded as one string, which the model cuts at 256 tokens (v1)."""
        return self._encoder.encode([text])[0]

    def pooled(self, text: str) -> np.ndarray:
        """Mean of the unit embeddings: a whole-document vector with nothing cut off."""
        mean = self.units(text).mean(axis=0)
        return mean / np.linalg.norm(mean)

    def skills(self, text: str) -> set[str]:
        return self._skills[text]

    def word_count(self, text: str) -> int:
        return len(text.split())


def interaction_features(resume: str, job: str, index: DocumentIndex) -> dict[str, float]:
    """Summary statistics of how one resume's units match one job's units."""
    similarity = index.units(resume) @ index.units(job).T
    best_per_requirement = similarity.max(axis=0)  # how well each job unit is covered
    best_per_statement = np.sort(similarity.max(axis=1))[::-1]  # each resume unit's best match
    resume_skills, job_skills = index.skills(resume), index.skills(job)
    shared = len(resume_skills & job_skills)

    return {
        "coverage_mean": best_per_requirement.mean(),
        "coverage_median": np.median(best_per_requirement),
        "coverage_min": best_per_requirement.min(),
        "coverage_strong_share": (best_per_requirement >= STRONG).mean(),
        "coverage_weak_share": (best_per_requirement < PARTIAL).mean(),
        "statement_top_mean": best_per_statement[:TOP_K].mean(),
        "statement_max": best_per_statement[0],
        "statement_mean": best_per_statement.mean(),
        "statement_relevant_share": (best_per_statement >= PARTIAL).mean(),
        "similarity_mean": similarity.mean(),
        "pooled_cosine": float(index.pooled(resume) @ index.pooled(job)),
        "whole_document_cosine": float(index.whole(resume) @ index.whole(job)),
        "skills_matched_share": shared / len(job_skills) if job_skills else 0.0,
        "skills_matched_count": shared,
        "skills_job_count": len(job_skills),
        "skills_resume_count": len(resume_skills),
        "skills_extra_count": len(resume_skills - job_skills),
        "resume_log_words": np.log1p(index.word_count(resume)),
        "job_log_words": np.log1p(index.word_count(job)),
    }


def pair_features(pairs: pd.DataFrame, index: DocumentIndex) -> pd.DataFrame:
    rows = [interaction_features(p.resume, p.job, index) for p in pairs.itertuples()]
    return pd.DataFrame(rows, dtype=float)


def embedding_features(pairs: pd.DataFrame, index: DocumentIndex) -> np.ndarray:
    """Element-wise product and absolute difference of the two pooled document vectors."""
    rows = []
    for pair in pairs.itertuples():
        resume, job = index.pooled(pair.resume), index.pooled(pair.job)
        rows.append(np.concatenate([resume * job, np.abs(resume - job)]))
    return np.array(rows, dtype=np.float32)
