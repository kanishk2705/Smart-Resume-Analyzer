import logging
import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

NLTK_PACKAGES = ("punkt_tab", "stopwords", "wordnet", "omw-1.4")


@lru_cache(maxsize=1)
def ensure_nltk_data() -> None:
    """Downloads any missing NLTK resources. Runs once per process."""
    for package in NLTK_PACKAGES:
        location = "tokenizers" if package == "punkt_tab" else "corpora"
        try:
            nltk.data.find(f"{location}/{package}")
        except LookupError:
            try:
                # Corpora are often left zipped, which find() only sees via the .zip path
                nltk.data.find(f"{location}/{package}.zip")
            except LookupError:
                logger.info("Downloading missing NLTK resource: %s", package)
                nltk.download(package, quiet=True)


@lru_cache(maxsize=1)
def _stop_words() -> frozenset[str]:
    ensure_nltk_data()
    return frozenset(stopwords.words("english"))


def split_sentences(text: str) -> list[str]:
    ensure_nltk_data()
    return nltk.sent_tokenize(text)


def clean_text(text: str) -> str:
    if not text:
        return ""

    ensure_nltk_data()

    # 1. Lowercase
    text = text.lower()

    # 2. Remove special chars (keep only alphabets)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # 3. Tokenization (Split into words)
    tokens = nltk.word_tokenize(text)

    # 4. Drop stopwords and lemmatize: "developing" -> "develop"
    lemmatizer = WordNetLemmatizer()
    stop_words = _stop_words()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return " ".join(clean_tokens)
