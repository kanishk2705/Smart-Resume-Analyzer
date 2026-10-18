from core.text import split_sentences, split_units


def test_split_sentences():
    assert split_sentences("I built a dashboard. It used SQL.") == [
        "I built a dashboard.",
        "It used SQL.",
    ]


def test_each_bullet_is_one_unit_with_its_marker_removed():
    text = "• Built dashboards in Tableau\n- Wrote SQL pipelines\n1. Trained junior analysts"

    assert split_units(text) == [
        "Built dashboards in Tableau",
        "Wrote SQL pipelines",
        "Trained junior analysts",
    ]


def test_short_bullets_are_not_split_on_abbreviations():
    assert split_units("B.Sc. in Statistics, University of Delhi") == [
        "B.Sc. in Statistics, University of Delhi"
    ]


def test_line_starting_lowercase_continues_the_previous_one():
    text = "• Built dashboards used by forty\nstore managers\n• Wrote SQL pipelines"

    assert split_units(text) == [
        "Built dashboards used by forty store managers",
        "Wrote SQL pipelines",
    ]


def test_long_line_without_sentence_end_is_treated_as_wrapped():
    first = "Automated the monthly revenue report for the finance team using a combination of"
    text = f"{first}\nPython and SQL\nData Analyst, Northwind"

    assert split_units(text) == [f"{first} Python and SQL", "Data Analyst, Northwind"]


def test_blank_line_ends_a_block():
    assert split_units("Data Analyst\n\nnorthwind retail") == ["Data Analyst", "northwind retail"]


def test_paragraphs_are_split_into_sentences():
    first = (
        "I am a data analyst with four years of experience in retail and logistics "
        "companies across three countries."
    )
    second = (
        "I build dashboards and pipelines that help operations teams make faster and "
        "better decisions every single week."
    )

    assert split_units(f"{first} {second}") == [first, second]


def test_overlong_units_are_cut_into_windows():
    units = split_units(" ".join(["word"] * 25), max_words=10)

    assert [len(unit.split()) for unit in units] == [10, 10, 5]


def test_empty_text_has_no_units():
    assert split_units("") == []
    assert split_units("\n  \n") == []
