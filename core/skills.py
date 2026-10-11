import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SKILLS_FILE = Path(__file__).parent / "data" / "skills.json"

# A skill must not be glued to other word characters: "java" must not match "javascript",
# and "c" must not match "c++" or "c#".
_BEFORE = r"(?<![A-Za-z0-9])"
_AFTER = r"(?![A-Za-z0-9+#])"

# Names such as "C", "R" or "Go" are ordinary words or initials in running text, so they
# only count when they stand alone in a delimited list ("Python, R, Go").
_LIST_BEFORE = r"(?:^|[,/|(:;•·]\s*)"
_LIST_AFTER = r"(?=\s*(?:[,/|);.]|$))"


@dataclass(frozen=True)
class Skill:
    name: str
    category: str
    # Skills in the same group are substitutes for one another, e.g. "relational database"
    group: str | None = None


def _alternation(terms: list[str]) -> str:
    # Longest first, so "react native" is tried before "react"
    longest_first = sorted(terms, key=len, reverse=True)
    return "|".join(re.escape(term).replace(r"\ ", r"\s+") for term in longest_first)


class SkillTaxonomy:
    """A dictionary of known skills, grouped by category, with the spellings each can take."""

    def __init__(self, data: dict[str, list[dict]]):
        self._matchers: list[tuple[Skill, list[re.Pattern]]] = []
        for category, entries in data.items():
            for entry in entries:
                exact = entry.get("exact", [])
                list_only = entry.get("list_only", [])
                aliases = list(entry.get("aliases", []))
                # The name itself matches in any case, unless it is listed as needing care
                if entry["name"] not in exact and entry["name"] not in list_only:
                    aliases.append(entry["name"])

                patterns = []
                if aliases:
                    patterns.append(
                        re.compile(_BEFORE + f"(?:{_alternation(aliases)})" + _AFTER, re.IGNORECASE)
                    )
                if exact:
                    patterns.append(re.compile(_BEFORE + f"(?:{_alternation(exact)})" + _AFTER))
                if list_only:
                    patterns.append(
                        re.compile(
                            _LIST_BEFORE + f"(?:{_alternation(list_only)})" + _LIST_AFTER,
                            re.MULTILINE,
                        )
                    )
                skill = Skill(entry["name"], category, entry.get("group"))
                self._matchers.append((skill, patterns))

    def extract(self, text: str) -> list[Skill]:
        """Returns the skills mentioned in the text, in taxonomy order."""
        return [
            skill
            for skill, patterns in self._matchers
            if any(pattern.search(text) for pattern in patterns)
        ]

    def __len__(self) -> int:
        return len(self._matchers)


@lru_cache(maxsize=1)
def get_taxonomy() -> SkillTaxonomy:
    return SkillTaxonomy(json.loads(SKILLS_FILE.read_text(encoding="utf-8")))
