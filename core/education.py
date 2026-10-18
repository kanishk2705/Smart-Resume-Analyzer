import re

LEVEL_NAMES = {1: "diploma", 2: "Bachelor's degree", 3: "Master's degree", 4: "doctorate"}

# Abbreviations are matched case-sensitively: "MS" is a degree, "ms" is milliseconds.
_MICROSOFT = r"(?!\s+(?:Office|Excel|Word|SQL|Access|PowerPoint|Project|Teams|Dynamics|Azure))"
_PATTERNS = {
    4: [
        re.compile(r"\bph\.?\s?d\b|\bdoctorate\b|\bdoctoral\b|\bd\.?phil\b", re.IGNORECASE),
    ],
    3: [
        # "master's", "masters", "master of/in" - but not "Scrum Master" or "master branch"
        re.compile(r"\bmaster(?:['’]?s\b|\s+(?:of|in|degree)\b)", re.IGNORECASE),
        re.compile(r"\bm\.?\s?tech\b|\bmba\b|\bm\.?eng\b|\bpost-?graduate\b", re.IGNORECASE),
        re.compile(r"\bM\.?Sc\b|\bM\.S\.|\bMS\b" + _MICROSOFT + r"|\bMCA\b|\bM\.A\."),
    ],
    2: [
        re.compile(
            r"\bbachelor(?:['’]?s)?\b|\bb\.?\s?tech\b|\bundergraduate degree\b", re.IGNORECASE
        ),
        re.compile(r"\bB\.?Sc\b|\bB\.S\.|\bBS\b|\bB\.E\.|\bBCA\b|\bB\.A\.|\bB\.?Com\b|\bBBA\b"),
        # A degree with no stated level is taken to mean a first degree
        re.compile(r"\b(?:university|college) degree\b|\bdegree in\b", re.IGNORECASE),
    ],
    1: [
        re.compile(r"\bdiploma\b|\bassociate(?:['’]?s)? degree\b", re.IGNORECASE),
    ],
}


def degree_levels(text: str) -> set[int]:
    """Every degree level mentioned in the text."""
    return {
        level
        for level, patterns in _PATTERNS.items()
        if any(pattern.search(text) for pattern in patterns)
    }


def highest_degree(text: str) -> int | None:
    """The highest degree level a resume mentions."""
    return max(degree_levels(text), default=None)


def required_degree(text: str) -> int | None:
    """The minimum degree a job description asks for.

    "Bachelor's or Master's" means a Bachelor's is enough, so the lowest level wins.
    """
    return min(degree_levels(text), default=None)


def education_score(resume_level: int | None, required_level: int | None) -> float | None:
    """100 if the requirement is met, 50 if one level short, otherwise 0.

    Returns None when the job description states no requirement.
    """
    if required_level is None:
        return None
    if resume_level is None:
        return 0.0
    if resume_level >= required_level:
        return 100.0
    return 50.0 if resume_level == required_level - 1 else 0.0
