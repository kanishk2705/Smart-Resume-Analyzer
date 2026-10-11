import pytest

from backend.storage import HistoryStore


def result(match_score=60.0, ats_score=80.0):
    return {"match_score": match_score, "skills": [], "ats": {"score": ats_score, "checks": []}}


@pytest.fixture
def store(tmp_path):
    return HistoryStore(str(tmp_path / "nested" / "history.db"))


def test_save_returns_result_with_id_and_metadata(store):
    saved = store.save("resume.pdf", "Data Analyst", result())

    assert saved["id"] == 1
    assert saved["resume_filename"] == "resume.pdf"
    assert saved["job_title"] == "Data Analyst"
    assert saved["created_at"].endswith("+00:00")
    assert saved["match_score"] == 60.0


def test_get_returns_the_full_result(store):
    saved = store.save("resume.pdf", "Data Analyst", result())

    fetched = store.get(saved["id"])

    assert fetched == saved
    assert store.get(999) is None


def test_list_is_newest_first_and_has_only_summaries(store):
    store.save("v1.pdf", "Data Analyst", result(50.0))
    store.save("v2.pdf", "Data Analyst", result(65.0))

    listed = store.list()

    assert [(a["resume_filename"], a["match_score"]) for a in listed] == [
        ("v2.pdf", 65.0),
        ("v1.pdf", 50.0),
    ]
    assert set(listed[0]) == {
        "id", "created_at", "resume_filename", "job_title", "match_score", "ats_score",
    }
    assert len(store.list(limit=1)) == 1


def test_delete(store):
    saved = store.save("resume.pdf", "Data Analyst", result())

    assert store.delete(saved["id"]) is True
    assert store.delete(saved["id"]) is False
    assert store.list() == []


def test_history_survives_reopening_the_file(tmp_path):
    path = str(tmp_path / "history.db")
    HistoryStore(path).save("resume.pdf", "Data Analyst", result())

    assert len(HistoryStore(path).list()) == 1
