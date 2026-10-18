import hashlib
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from core.sections import RESUME_HEADINGS

DATASET = "cnamuangtoun/resume-job-description-fit"
# Ordinal: a higher number is a better fit
LABELS = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}

# The dataset's text has lost most of its line breaks: sentences are glued together
# ("...consulting projects.Proven ability...") and so are headings ("Summary7+ years").
_GLUED_SENTENCE = re.compile(r"(?<=[a-z0-9)])([.!?:])(?=[A-Z])")
_HEADING_WORDS = sorted(
    {alias.title() for aliases in RESUME_HEADINGS.values() for alias in aliases},
    key=len,
    reverse=True,
)
_GLUED_HEADING = re.compile(
    r"^(" + "|".join(re.escape(h) for h in _HEADING_WORDS) + r")(?=[A-Z0-9])", re.MULTILINE
)


def restore_line_breaks(text: str) -> str:
    """Puts back line breaks after glued sentences and glued section headings."""
    text = _GLUED_SENTENCE.sub(r"\1\n", text)
    return _GLUED_HEADING.sub(r"\1\n", text)


def resume_fold(resume: str, folds: int) -> int:
    """Assigns a resume to one of `folds` groups, the same way on every run and machine."""
    return int(hashlib.sha1(resume.encode("utf-8")).hexdigest()[:8], 16) % folds


def load_split(split: str, cache_dir: Path) -> pd.DataFrame:
    """Downloads one split ("train" or "test") and returns resume, job and ordinal label."""
    path = hf_hub_download(DATASET, f"{split}.csv", repo_type="dataset", local_dir=cache_dir)
    raw = pd.read_csv(path)
    return pd.DataFrame({
        "resume": raw["resume_text"].map(restore_line_breaks),
        "job": raw["job_description_text"].map(restore_line_breaks),
        "label": raw["label"].map(LABELS),
    })
