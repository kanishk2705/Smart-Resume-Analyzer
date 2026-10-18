import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

import numpy as np

from core.config import Settings, get_settings
from core.education import LEVEL_NAMES, education_score, highest_degree, required_degree
from core.sections import HEADER, JD_HEADINGS, split_sections
from core.skills import SkillTaxonomy, get_taxonomy
from core.text import split_units

logger = logging.getLogger(__name__)

# Units this short are usually headings or page furniture ("Page 1", "Skills")
MIN_UNIT_WORDS = 4

EXPERIENCE_SECTIONS = ("experience", "projects")
# Parts of a job description that describe the employer, not the candidate
JD_IGNORED_SECTIONS = ("company", "benefits")


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
        return self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )


@dataclass
class ScoredSentence:
    text: str
    score: float


@dataclass
class ScoreComponent:
    key: str
    label: str
    score: float | None  # 0-100, or None when it could not be assessed
    weight: float
    detail: str


@dataclass
class SkillMatch:
    name: str
    category: str
    status: str  # "matched" | "related" | "missing"
    related_to: str | None = None


@dataclass
class RequirementMatch:
    requirement: str
    evidence: str
    similarity: float
    status: str  # "strong" | "partial" | "weak"


@dataclass
class AnalysisResult:
    match_score: float
    components: list[ScoreComponent] = field(default_factory=list)
    skills: list[SkillMatch] = field(default_factory=list)
    additional_skills: list[str] = field(default_factory=list)
    requirements: list[RequirementMatch] = field(default_factory=list)
    top_sentences: list[ScoredSentence] = field(default_factory=list)
    sections_found: list[str] = field(default_factory=list)


class ResumeAnalyzer:
    def __init__(
        self,
        encoder: Encoder,
        taxonomy: SkillTaxonomy | None = None,
        settings: Settings | None = None,
    ):
        self._encoder = encoder
        self._taxonomy = taxonomy or get_taxonomy()
        self._settings = settings or Settings()

    def analyze(self, resume_text: str, jd_text: str) -> AnalysisResult:
        sections = split_sections(resume_text)
        resume_units, unit_sections = self.resume_units(resume_text, sections)
        jd_units = self.jd_units(jd_text)

        # One embedding per sentence / bullet, so nothing is lost to the model's length limit.
        # similarity[i, j] compares resume unit i with job description unit j.
        embeddings = self._encoder.encode(resume_units + jd_units)
        similarity = embeddings[: len(resume_units)] @ embeddings[len(resume_units) :].T

        skills, additional_skills = self._match_skills(resume_text, jd_text)
        education_text = next((s.text for s in sections if s.name == "education"), resume_text)

        components = [
            self._skills_component(skills),
            self._coverage_component(similarity),
            self._experience_component(similarity, unit_sections),
            self._education_component(education_text, jd_text),
        ]

        return AnalysisResult(
            match_score=_weighted_mean(components),
            components=components,
            skills=skills,
            additional_skills=additional_skills,
            requirements=self._requirements(similarity, resume_units, jd_units),
            top_sentences=self._top_sentences(similarity, resume_units),
            sections_found=[s.name for s in sections if s.name != HEADER],
        )

    # --- Splitting ---

    def resume_units(self, resume_text, sections) -> tuple[list[str], list[str]]:
        """Resume sentences, and the section each one came from."""
        pairs = [
            (unit, section.name)
            for section in sections
            for unit in split_units(section.text, self._settings.max_unit_words)
        ]
        long_enough = [pair for pair in pairs if len(pair[0].split()) >= MIN_UNIT_WORDS]
        # A resume made only of fragments still needs something to compare
        pairs = long_enough or pairs or [(resume_text.strip(), HEADER)]
        return [unit for unit, _ in pairs], [name for _, name in pairs]

    def jd_units(self, jd_text: str) -> list[str]:
        """Job description sentences, without "about us" and benefits sections."""
        relevant = [
            section.text
            for section in split_sections(jd_text, JD_HEADINGS)
            if section.name not in JD_IGNORED_SECTIONS
        ]
        units = split_units("\n".join(relevant) or jd_text, self._settings.max_unit_words)
        return [u for u in units if len(u.split()) >= MIN_UNIT_WORDS] or units or [jd_text.strip()]

    # --- Skills ---

    def _match_skills(self, resume_text: str, jd_text: str) -> tuple[list[SkillMatch], list[str]]:
        """Classifies every skill the JD asks for as matched, related or missing."""
        jd_skills = self._taxonomy.extract(jd_text)
        resume_skills = self._taxonomy.extract(resume_text)
        resume_names = {skill.name for skill in resume_skills}
        additional = [skill.name for skill in resume_skills if skill not in jd_skills]

        # A substitute the candidate does have, e.g. MySQL when the job asks for PostgreSQL
        substitutes: dict[str, str] = {}
        for skill in resume_skills:
            if skill.group:
                substitutes.setdefault(skill.group, skill.name)

        matches = []
        for skill in jd_skills:
            if skill.name in resume_names:
                matches.append(SkillMatch(skill.name, skill.category, "matched"))
            elif skill.group in substitutes:
                matches.append(
                    SkillMatch(skill.name, skill.category, "related", substitutes[skill.group])
                )
            else:
                matches.append(SkillMatch(skill.name, skill.category, "missing"))
        return matches, additional

    # --- Score components ---

    def _skills_component(self, skills: list[SkillMatch]) -> ScoreComponent:
        weight = self._settings.weight_skills
        if not skills:
            return ScoreComponent(
                "skills",
                "Skills match",
                None,
                weight,
                "No known skills found in the job description.",
            )

        matched = sum(s.status == "matched" for s in skills)
        related = sum(s.status == "related" for s in skills)
        # A related skill earns half credit
        score = (matched + 0.5 * related) / len(skills) * 100
        detail = f"{matched} of {len(skills)} required skills found"
        if related:
            detail += f", {related} more covered by a related skill"
        return ScoreComponent("skills", "Skills match", round(score, 2), weight, detail + ".")

    def _coverage_component(self, similarity: np.ndarray) -> ScoreComponent:
        """How much of what the job asks for is evidenced somewhere in the resume."""
        best_per_requirement = similarity.max(axis=0)
        strong = int((best_per_requirement >= self._settings.strong_match_threshold).sum())
        return ScoreComponent(
            "coverage",
            "Requirement coverage",
            self._rescale(best_per_requirement.mean()),
            self._settings.weight_coverage,
            f"{strong} of {len(best_per_requirement)} job requirements have strong evidence "
            "in the resume.",
        )

    def _experience_component(
        self, similarity: np.ndarray, unit_sections: list[str]
    ) -> ScoreComponent:
        """How relevant the candidate's most relevant experience is to this job."""
        weight = self._settings.weight_experience
        rows = [i for i, name in enumerate(unit_sections) if name in EXPERIENCE_SECTIONS]
        if not rows:
            return ScoreComponent(
                "experience",
                "Experience relevance",
                None,
                weight,
                "No Experience or Projects section found in the resume.",
            )

        # Judged on the best few bullets, so a long and varied history is not penalised
        best_per_bullet = np.sort(similarity[rows].max(axis=1))[::-1]
        top = best_per_bullet[: self._settings.top_experience_units]
        return ScoreComponent(
            "experience",
            "Experience relevance",
            self._rescale(top.mean()),
            weight,
            f"Based on the {len(top)} most relevant of {len(rows)} experience and project "
            "statements.",
        )

    def _education_component(self, education_text: str, jd_text: str) -> ScoreComponent:
        weight = self._settings.weight_education
        required = required_degree(jd_text)
        held = highest_degree(education_text)
        score = education_score(held, required)

        if required is None:
            detail = "The job description does not state a degree requirement."
        elif held is None:
            detail = f"Job asks for a {LEVEL_NAMES[required]}; no degree found in the resume."
        else:
            detail = (
                f"Job asks for a {LEVEL_NAMES[required]}; "
                f"resume shows a {LEVEL_NAMES[held]}."
            )
        return ScoreComponent("education", "Education", score, weight, detail)

    def _rescale(self, similarity: float) -> float:
        """Maps a raw cosine similarity onto 0-100.

        Unrelated sentences rarely score 0 and near-paraphrases rarely score 1, so the
        raw value is stretched between a floor and a ceiling set in the configuration.
        """
        floor, ceiling = self._settings.similarity_floor, self._settings.similarity_ceiling
        scaled = (float(similarity) - floor) / (ceiling - floor)
        return round(min(max(scaled, 0.0), 1.0) * 100, 2)

    # --- Explanations ---

    def _requirements(self, similarity, resume_units, jd_units) -> list[RequirementMatch]:
        """For each job requirement, the resume sentence that supports it best."""
        matches = []
        for j, requirement in enumerate(jd_units):
            best = int(similarity[:, j].argmax())
            score = float(similarity[best, j])
            if score >= self._settings.strong_match_threshold:
                status = "strong"
            elif score >= self._settings.partial_match_threshold:
                status = "partial"
            else:
                status = "weak"
            matches.append(RequirementMatch(requirement, resume_units[best], score, status))
        return matches

    def _top_sentences(self, similarity, resume_units) -> list[ScoredSentence]:
        """Explainability: the resume sentences that best match any job requirement."""
        best_per_unit = similarity.max(axis=1)
        ranked = np.argsort(-best_per_unit, kind="stable")[: self._settings.top_k_sentences]
        return [ScoredSentence(resume_units[i], float(best_per_unit[i])) for i in ranked]


def _weighted_mean(components: list[ScoreComponent]) -> float:
    """Overall score. Components that could not be assessed are left out, not counted as 0."""
    assessed = [c for c in components if c.score is not None and c.weight > 0]
    total_weight = sum(c.weight for c in assessed)
    if not total_weight:
        return 0.0
    return round(sum(c.score * c.weight for c in assessed) / total_weight, 2)


@lru_cache(maxsize=1)
def get_analyzer() -> ResumeAnalyzer:
    """The process-wide analyzer. The model loads once, on first call."""
    settings = get_settings()
    return ResumeAnalyzer(
        encoder=SentenceTransformerEncoder(settings.embedding_model),
        settings=settings,
    )
