"""Format checks: will an applicant tracking system read this resume correctly?

These checks look only at the resume, not at any job description.
"""

import re
from dataclasses import dataclass, field

from core.parsing import ParsedDocument
from core.sections import Section, split_sections
from core.text import bullet_points, split_units

PASS, WARN, FAIL = "pass", "warn", "fail"
_CREDIT = {PASS: 1.0, WARN: 0.5, FAIL: 0.0}

MAX_EXAMPLES = 3
MIN_WORDS, MAX_WORDS, MAX_PAGES = 200, 1000, 2
MAX_BULLET_WORDS = 40
# Share of bullets that should contain a number
QUANTIFIED_GOOD, QUANTIFIED_LOW = 0.5, 0.25

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_LIKE = re.compile(r"(?<!\d)\+?\d[\d ().-]{8,18}\d(?!\d)")
_PROFILE = re.compile(r"linkedin\.com|github\.com", re.IGNORECASE)
# Icon fonts and unmapped glyphs come out of PDF extraction as private-use or
# replacement characters, or as "(cid:123)"
_UNREADABLE = re.compile(r"[-�]|\(cid:\d+\)")
_NUMBER = re.compile(r"\d")
_PRONOUN = re.compile(r"\bI\b(?!\.)|\b[Mm]y\b|\b[Mm]e\b")
_WEAK_OPENERS = (
    "responsible for", "worked on", "worked with", "helped", "assisted", "duties included",
    "tasked with", "participated in", "involved in", "was part of", "handled",
)
_BULLET_SECTIONS = ("experience", "projects")
_REQUIRED_SECTIONS = {
    "experience or projects": ("experience", "projects"),
    "education": ("education",),
    "skills": ("skills",),
}


@dataclass
class AtsCheck:
    key: str
    label: str
    status: str  # "pass" | "warn" | "fail"
    message: str
    examples: list[str] = field(default_factory=list)


@dataclass
class AtsReport:
    score: float  # 0-100: a pass earns full credit, a warning half
    checks: list[AtsCheck]


def check_resume(document: ParsedDocument) -> AtsReport:
    sections = split_sections(document.text)
    bullets = _bullets(sections)
    checks = [
        _email(document),
        _phone(document),
        _profile_links(document),
        _sections(sections),
        _length(document),
        _unreadable_characters(document.text),
        _quantified(bullets),
        _weak_openers(bullets),
        _long_bullets(bullets),
        _pronouns(sections),
    ]
    score = sum(_CREDIT[check.status] for check in checks) / len(checks) * 100
    return AtsReport(round(score, 2), checks)


def _bullets(sections: list[Section]) -> list[str]:
    """Achievement statements: bullets under Experience / Projects where those exist."""
    relevant = [s.text for s in sections if s.name in _BULLET_SECTIONS]
    text = "\n".join(relevant or [s.text for s in sections])
    # A resume without bullet markers is judged on its sentences instead
    return bullet_points(text) or [u for u in split_units(text) if len(u.split()) >= 8]


def _contact(label, key, in_text: bool, link_prefix, document) -> AtsCheck:
    if in_text:
        return AtsCheck(key, label, PASS, f"{label} found.")
    if any(link.lower().startswith(link_prefix) for link in document.links):
        return AtsCheck(
            key,
            label,
            WARN,
            f"{label} exists only as a hyperlink, not as visible text. Many ATS parsers "
            "read text only, so write it out in full.",
        )
    return AtsCheck(key, label, FAIL, f"No {label.lower()} found. Recruiters cannot reach you.")


def _email(document: ParsedDocument) -> AtsCheck:
    in_text = bool(_EMAIL.search(document.text))
    return _contact("Email address", "email", in_text, "mailto:", document)


def _phone(document: ParsedDocument) -> AtsCheck:
    # 10-15 digits, so that a date range such as "2019 - 2021" is not taken for a phone
    in_text = any(
        10 <= sum(c.isdigit() for c in candidate) <= 15
        for candidate in _PHONE_LIKE.findall(document.text)
    )
    return _contact("Phone number", "phone", in_text, "tel:", document)


def _profile_links(document: ParsedDocument) -> AtsCheck:
    label = "LinkedIn / GitHub"
    if _PROFILE.search(document.text):
        return AtsCheck("profiles", label, PASS, "Profile URL found.")
    if any(_PROFILE.search(link) for link in document.links):
        return AtsCheck(
            "profiles",
            label,
            WARN,
            "Profile links exist only as hyperlinks. Write the URL out so it survives "
            "text-only parsing and printing.",
        )
    return AtsCheck("profiles", label, WARN, "No LinkedIn or GitHub URL found.")


def _sections(sections: list[Section]) -> AtsCheck:
    found = {section.name for section in sections}
    missing = [
        label for label, names in _REQUIRED_SECTIONS.items() if not found.intersection(names)
    ]
    if missing:
        return AtsCheck(
            "sections",
            "Standard sections",
            FAIL,
            f"No recognisable heading for: {', '.join(missing)}. Use conventional headings "
            'such as "Experience", "Education" and "Skills".',
        )
    return AtsCheck("sections", "Standard sections", PASS, "All essential sections found.")


def _length(document: ParsedDocument) -> AtsCheck:
    words = len(document.text.split())
    pages = document.page_count
    size = f"{words} words" + (f" on {pages} page{'s' if pages != 1 else ''}" if pages else "")
    if pages and pages > MAX_PAGES:
        return AtsCheck("length", "Length", WARN, f"{size}. Aim for one or two pages.")
    if words < MIN_WORDS:
        return AtsCheck("length", "Length", WARN, f"{size}. This is thin; add detail.")
    if words > MAX_WORDS:
        return AtsCheck("length", "Length", WARN, f"{size}. This is long; cut older detail.")
    return AtsCheck("length", "Length", PASS, f"{size}.")


def _unreadable_characters(text: str) -> AtsCheck:
    count = len(_UNREADABLE.findall(text))
    if count:
        return AtsCheck(
            "characters",
            "Readable characters",
            WARN,
            f"{count} symbol{'s' if count != 1 else ''} could not be read as text. "
            "These are usually icons; an ATS sees junk in their place. Use plain text labels.",
        )
    return AtsCheck("characters", "Readable characters", PASS, "All text extracted cleanly.")


def _quantified(bullets: list[str]) -> AtsCheck:
    label = "Quantified achievements"
    if not bullets:
        return AtsCheck("quantified", label, WARN, "No achievement statements found.")

    without = [b for b in bullets if not _NUMBER.search(b)]
    share = 1 - len(without) / len(bullets)
    message = f"{len(bullets) - len(without)} of {len(bullets)} statements include a number."
    if share >= QUANTIFIED_GOOD:
        return AtsCheck("quantified", label, PASS, message)
    status = WARN if share >= QUANTIFIED_LOW else FAIL
    return AtsCheck(
        "quantified",
        label,
        status,
        message + " Add a measurable result (how much, how many, how fast) where you can.",
        without[:MAX_EXAMPLES],
    )


def _weak_openers(bullets: list[str]) -> AtsCheck:
    weak = [b for b in bullets if b.lower().startswith(_WEAK_OPENERS)]
    if weak:
        return AtsCheck(
            "verbs",
            "Action verbs",
            WARN,
            f"{len(weak)} statement{'s' if len(weak) != 1 else ''} open with a passive "
            'phrase. Start with what you did: "Built", "Led", "Reduced".',
            weak[:MAX_EXAMPLES],
        )
    return AtsCheck("verbs", "Action verbs", PASS, "No statements open with a passive phrase.")


def _long_bullets(bullets: list[str]) -> AtsCheck:
    long = [b for b in bullets if len(b.split()) > MAX_BULLET_WORDS]
    if long:
        return AtsCheck(
            "bullet_length",
            "Bullet length",
            WARN,
            f"{len(long)} statement{'s' if len(long) != 1 else ''} run past "
            f"{MAX_BULLET_WORDS} words. Split or trim them.",
            long[:MAX_EXAMPLES],
        )
    return AtsCheck("bullet_length", "Bullet length", PASS, "All statements are concise.")


def _pronouns(sections: list[Section]) -> AtsCheck:
    units = [u for section in sections for u in split_units(section.text)]
    with_pronoun = [u for u in units if _PRONOUN.search(u)]
    if with_pronoun:
        return AtsCheck(
            "pronouns",
            "First-person pronouns",
            WARN,
            'Resumes conventionally drop "I", "my" and "me".',
            with_pronoun[:MAX_EXAMPLES],
        )
    return AtsCheck("pronouns", "First-person pronouns", PASS, "None found.")
