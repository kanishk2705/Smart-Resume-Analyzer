"""LLM-written advice: what to learn for missing skills, and how to rephrase weak bullets."""

import urllib.parse

from core.config import get_settings
from core.llm import complete_json

_LEARNING_SYSTEM = """You are a career coach helping a candidate prepare for a specific job.
For each skill you are given, recommend:
- "resource_name": one specific, well-known learning resource, with its provider
  (for example "Coursera: Python for Everybody"). Do not include a URL.
- "project": one small portfolio project, in a single sentence, that would demonstrate
  the skill to an employer hiring for this job.

Respond with JSON only, in this shape:
{"recommendations": [{"skill": "...", "resource_name": "...", "project": "..."}]}
Include exactly one entry per skill, in the order given."""

_REWRITE_SYSTEM = """You are an expert resume editor. Each resume statement you are given was
flagged by a reviewer: it has no measurable result, opens weakly, runs too long or uses
first-person pronouns. Rewrite each one so that it is stronger for the target job. Every
rewrite must differ from its original.

Rules:
- Start with a strong past-tense action verb. Never use "I", "my" or "responsible for".
- Keep it to one sentence of at most 30 words.
- NEVER invent facts, tools, numbers or results that are not in the original.
- Where a measurable result would strengthen the statement but none is given, insert a
  placeholder in square brackets for the candidate to fill in, such as [X%] or [N users].
- "note" says, in at most 12 words, what you changed or what the candidate should add.

Respond with JSON only, in this shape:
{"rewrites": [{"original": "...", "rewritten": "...", "note": "..."}]}
Include exactly one entry per statement, in the order given."""


def _entries(answer: dict, key: str, fields: tuple[str, ...]) -> list[dict]:
    """The well-formed entries of an LLM answer, with every field coerced to a string."""
    items = answer.get(key)
    if not isinstance(items, list):
        return []
    return [
        {field: str(item.get(field) or "") for field in fields}
        for item in items
        if isinstance(item, dict)
    ]


def learning_path(missing_skills: list[str], job_title: str) -> list[dict]:
    """A learning resource and a project idea for each of the top missing skills."""
    skills = missing_skills[: get_settings().max_coached_items]
    if not skills:
        return []

    answer = complete_json(
        _LEARNING_SYSTEM, f"Target job: {job_title}\nSkills to learn: {', '.join(skills)}"
    )
    recommendations = _entries(answer, "recommendations", ("skill", "resource_name", "project"))
    # LLM-written URLs are often invented, so we link to a search for the resource instead
    for rec in recommendations:
        query = urllib.parse.quote(rec["resource_name"] or rec["skill"])
        rec["link"] = f"https://www.google.com/search?q={query}"
    return recommendations


def rewrite_bullets(bullets: list[str], job_title: str) -> list[dict]:
    """Stronger wordings of resume statements, without adding facts that were not there."""
    bullets = bullets[: get_settings().max_coached_items]
    if not bullets:
        return []

    numbered = "\n".join(f"{i}. {bullet}" for i, bullet in enumerate(bullets, start=1))
    answer = complete_json(_REWRITE_SYSTEM, f"Target job: {job_title}\nStatements:\n{numbered}")
    rewrites = _entries(answer, "rewrites", ("original", "rewritten", "note"))
    # The model may paraphrase the original when echoing it back; show the real one
    for rewrite, bullet in zip(rewrites, bullets, strict=False):
        rewrite["original"] = bullet
    return [r for r in rewrites if r["rewritten"] and r["rewritten"] != r["original"]]
