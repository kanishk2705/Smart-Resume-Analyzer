from core.cleaning import clean_text, split_sentences


def test_empty_input_returns_empty_string():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_lowercases_and_strips_punctuation_and_digits():
    assert clean_text("Python, SQL & Docker (2024)!") == "python sql docker"


def test_removes_stopwords():
    assert clean_text("I am the owner of a dashboard") == "owner dashboard"


def test_lemmatizes_plurals():
    assert clean_text("built dashboards and pipelines") == "built dashboard pipeline"


def test_split_sentences():
    assert split_sentences("I built a dashboard. It used SQL.") == [
        "I built a dashboard.",
        "It used SQL.",
    ]
