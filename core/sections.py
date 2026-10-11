import re
from dataclasses import dataclass

# Text before the first recognised heading (usually name and contact details)
HEADER = "header"

RESUME_HEADINGS = {
    "summary": [
        "summary", "professional summary", "career summary", "profile", "professional profile",
        "objective", "career objective", "about me", "about",
    ],
    "experience": [
        "experience", "work experience", "professional experience", "relevant experience",
        "employment", "employment history", "work history", "internships", "internship",
        "internship experience", "research experience", "industry experience",
    ],
    "education": [
        "education", "academic background", "academic qualifications", "academics",
        "educational qualifications", "education and training", "qualifications",
    ],
    "skills": [
        "skills", "technical skills", "key skills", "core skills", "core competencies",
        "competencies", "skills and tools", "tools and technologies", "technologies",
        "technical proficiencies", "skills and technologies", "areas of expertise",
    ],
    "projects": [
        "projects", "academic projects", "personal projects", "key projects",
        "selected projects", "project experience", "research projects",
    ],
    "certifications": [
        "certifications", "certificates", "licenses and certifications", "courses",
        "certifications and courses", "training", "coursework", "relevant coursework",
    ],
    "other": [
        "awards", "achievements", "honors", "honors and awards", "awards and achievements",
        "publications", "languages", "interests", "hobbies", "volunteer experience",
        "volunteering", "extracurricular activities", "activities", "leadership",
        "positions of responsibility", "references",
    ],
}

JD_HEADINGS = {
    "requirements": [
        "requirements", "qualifications", "required qualifications", "minimum qualifications",
        "preferred qualifications", "basic qualifications", "what you bring",
        "what you will bring", "what we are looking for", "who you are", "skills",
        "required skills", "skills and qualifications", "must have", "nice to have",
        "about you", "your profile", "experience",
    ],
    "responsibilities": [
        "responsibilities", "key responsibilities", "what you will do", "what you ll do",
        "the role", "about the role", "role overview", "job description", "duties",
        "your responsibilities", "day to day", "role and responsibilities",
    ],
    "company": [
        "about us", "about the company", "who we are", "company overview", "our company",
        "our mission", "about the team",
    ],
    "benefits": [
        "benefits", "perks", "perks and benefits", "what we offer", "why join us",
        "compensation", "compensation and benefits", "salary", "how to apply",
        "equal opportunity", "equal opportunity employer", "diversity and inclusion",
    ],
}

MAX_HEADING_WORDS = 5


@dataclass
class Section:
    name: str
    text: str


def _normalise_heading(line: str) -> str:
    line = line.strip()
    # Letter-spaced headings from designed PDFs: "E D U C A T I O N"
    if re.fullmatch(r"(?:\S\s)+\S", line):
        line = line.replace(" ", "")
    line = line.lower().replace("&", " and ").replace("’", " ").replace("'", " ")
    return " ".join(re.sub(r"[^a-z\s]", " ", line).split())


def split_sections(text: str, headings: dict[str, list[str]] = RESUME_HEADINGS) -> list[Section]:
    """Splits a document at lines that are known section headings.

    Sections come back in document order. Repeated headings of the same kind
    (e.g. "Experience" and "Internships") are merged into one section.
    """
    lookup = {alias: name for name, aliases in headings.items() for alias in aliases}

    lines_by_section: dict[str, list[str]] = {HEADER: []}
    current = HEADER
    for line in text.splitlines():
        candidate = _normalise_heading(line)
        if candidate in lookup and len(candidate.split()) <= MAX_HEADING_WORDS:
            current = lookup[candidate]
            lines_by_section.setdefault(current, [])
        else:
            lines_by_section[current].append(line)

    sections = [Section(name, "\n".join(lines).strip()) for name, lines in lines_by_section.items()]
    return [section for section in sections if section.text]
