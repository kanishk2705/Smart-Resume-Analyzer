import logging
import re
from functools import lru_cache

import nltk

logger = logging.getLogger(__name__)

_BULLET = re.compile(r"^(?:[•●▪■◦‣·*\-–—o]\s+|\d{1,2}[.)]\s+)")
_SENTENCE_END = (".", "!", "?", ":", ";")

# PDF extraction breaks a long sentence at the page margin. A line this long that doesn't
# end a sentence is assumed to continue on the next line.
WRAPPED_LINE_CHARS = 70
PARAGRAPH_WORDS = 30


@lru_cache(maxsize=1)
def ensure_nltk_data() -> None:
    """Downloads the sentence tokenizer if it is missing. Runs once per process."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        logger.info("Downloading missing NLTK resource: punkt_tab")
        nltk.download("punkt_tab", quiet=True)


def split_sentences(text: str) -> list[str]:
    ensure_nltk_data()
    return nltk.sent_tokenize(text)


def _split_blocks(text: str) -> list[tuple[str, bool]]:
    """Groups lines into bullets / paragraphs, re-joining lines that were wrapped.

    Returns each block with whether it began with a bullet marker.
    """
    blocks: list[tuple[str, bool]] = []
    previous = ""  # the line before this one; empty after a blank line
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            previous = ""
            continue

        is_bullet = bool(_BULLET.match(line))
        line = _BULLET.sub("", line).strip()
        if not line:
            continue

        previous_was_wrapped = (
            len(previous) >= WRAPPED_LINE_CHARS and not previous.endswith(_SENTENCE_END)
        )
        continues = previous and not is_bullet and (line[0].islower() or previous_was_wrapped)
        if blocks and continues:
            blocks[-1] = (blocks[-1][0] + " " + line, blocks[-1][1])
        else:
            blocks.append((line, is_bullet))
        previous = line
    return blocks


def bullet_points(text: str) -> list[str]:
    """The bullet points in a document, without their markers."""
    return [block for block, is_bullet in _split_blocks(text) if is_bullet]


def split_units(text: str, max_words: int = 120) -> list[str]:
    """Splits a document into sentence-sized units, one per bullet point or sentence.

    Units longer than `max_words` are cut into consecutive windows, so that no unit
    exceeds what the embedding model can read.
    """
    units = []
    for block, _ in _split_blocks(text):
        # A short block is one bullet point. Splitting it further would only break on
        # abbreviations ("B.Sc. in Statistics"), so only paragraphs are split.
        is_paragraph = len(block.split()) > PARAGRAPH_WORDS
        for sentence in split_sentences(block) if is_paragraph else [block]:
            words = sentence.split()
            for start in range(0, len(words), max_words):
                units.append(" ".join(words[start : start + max_words]))
    return units
