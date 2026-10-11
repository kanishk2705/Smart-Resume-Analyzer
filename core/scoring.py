import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from core.cleaning import clean_text, split_sentences
from core.config import get_settings

logger = logging.getLogger(__name__)

# Sentences this short are usually headings or page furniture ("Page 1", "Skills")
MIN_SENTENCE_WORDS = 5


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Returns one L2-normalised embedding per text, shape (len(texts), dim)."""
        ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str):
        # Imported here so the heavy torch import only happens when a real model is needed
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model '%s'", model_name)
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


@dataclass
class ScoredSentence:
    text: str
    score: float


@dataclass
class AnalysisResult:
    match_score: float
    missing_keywords: list[str] = field(default_factory=list)
    top_sentences: list[ScoredSentence] = field(default_factory=list)


class ResumeAnalyzer:
    def __init__(self, encoder: Encoder, top_k_sentences: int = 3, max_missing_keywords: int = 10):
        self._encoder = encoder
        self._top_k_sentences = top_k_sentences
        self._max_missing_keywords = max_missing_keywords

    def analyze(self, resume_text: str, jd_text: str) -> AnalysisResult:
        # --- PART 1: OVERALL SEMANTIC SCORE ---
        # We use the RAW text because BERT needs grammar to understand context
        resume_embedding, jd_embedding = self._encoder.encode([resume_text, jd_text])
        similarity_score = float(resume_embedding @ jd_embedding)
        match_percentage = round(similarity_score * 100, 2)

        return AnalysisResult(
            match_score=match_percentage,
            missing_keywords=self._missing_keywords(resume_text, jd_text),
            top_sentences=self._top_sentences(resume_text, jd_embedding),
        )

    def _top_sentences(self, resume_text: str, jd_embedding: np.ndarray) -> list[ScoredSentence]:
        """Explainability: the resume sentences that best match the JD."""
        sentences = split_sentences(resume_text)
        valid_sentences = [s for s in sentences if len(s.split()) >= MIN_SENTENCE_WORDS]
        if not valid_sentences:
            return []

        scores = self._encoder.encode(valid_sentences) @ jd_embedding
        ranked = sorted(
            zip(valid_sentences, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        return [ScoredSentence(text=s, score=float(v)) for s, v in ranked[: self._top_k_sentences]]

    def _missing_keywords(self, resume_text: str, jd_text: str) -> list[str]:
        """Keyword gap: terms in the JD but not in the resume, most frequent first."""
        # We use the CLEAN text here to match specific skills (e.g., "Python")
        cv = CountVectorizer(stop_words="english", ngram_range=(1, 2))
        try:
            counts = cv.fit_transform([clean_text(resume_text), clean_text(jd_text)]).toarray()
        except ValueError:
            # Happens if documents are empty or contain only stop words
            return []

        resume_counts, jd_counts = counts
        terms = cv.get_feature_names_out()
        missing = np.flatnonzero((jd_counts > 0) & (resume_counts == 0))
        ranked = sorted(missing, key=lambda i: jd_counts[i], reverse=True)
        return [str(terms[i]) for i in ranked[: self._max_missing_keywords]]


@lru_cache(maxsize=1)
def get_analyzer() -> ResumeAnalyzer:
    """The process-wide analyzer. The model loads once, on first call."""
    settings = get_settings()
    return ResumeAnalyzer(
        encoder=SentenceTransformerEncoder(settings.embedding_model),
        top_k_sentences=settings.top_k_sentences,
        max_missing_keywords=settings.max_missing_keywords,
    )
