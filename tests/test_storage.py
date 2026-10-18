import pytest

from backend.storage import HistoryStore


def result(match_score=60.0, ats_score=80.0):
    return {"match_score": match_score, "skills": [], "ats": {"score": ats_score, "checks": []}}


@pytest.fixture
def store(tmp_path):
    return HistoryStore(str(tmp_path / "nested" / "history.db"))


def test_save_returns_result_with_id_and_metadata(store):
    saved = store.save("resume.pdf", "Data Analyst", result(), owner=None)

    assert saved["id"] == 1
    assert saved["resume_filename"] == "resume.pdf"
    assert saved["job_title"] == "Data Analyst"
    assert saved["created_at"].endswith("+00:00")
    assert saved["match_score"] == 60.0


def test_get_returns_the_full_result(store):
    saved = store.save("resume.pdf", "Data Analyst", result(), owner=None)

    fetched = store.get(saved["id"])

    assert fetched == saved
    assert store.get(999) is None


def test_list_is_newest_first_and_has_only_summaries(store):
    store.save("v1.pdf", "Data Analyst", result(50.0), owner=None)
    store.save("v2.pdf", "Data Analyst", result(65.0), owner=None)

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
    saved = store.save("resume.pdf", "Data Analyst", result(), owner=None)

    assert store.delete(saved["id"]) is True
    assert store.delete(saved["id"]) is False
    assert store.list() == []


def test_history_survives_reopening_the_file(tmp_path):
    path = str(tmp_path / "history.db")
    HistoryStore(path).save("resume.pdf", "Data Analyst", result(), owner=None)

    assert len(HistoryStore(path).list()) == 1


def test_each_save_gets_a_distinct_unguessable_share_token(store):
    first = store.save("a.pdf", "Data Analyst", result(), owner=None)
    second = store.save("b.pdf", "Data Analyst", result(), owner=None)

    assert first["share_token"] != second["share_token"]
    assert len(first["share_token"]) >= 20
    assert store.get_shared(first["share_token"])["resume_filename"] == "a.pdf"
    assert store.get_shared("not-a-real-token") is None


def test_owner_limits_what_can_be_listed_fetched_and_deleted(store):
    mine = store.save("mine.pdf", "Data Analyst", result(), owner="session-a")
    theirs = store.save("theirs.pdf", "Data Analyst", result(), owner="session-b")

    assert [a["resume_filename"] for a in store.list(owner="session-a")] == ["mine.pdf"]
    assert store.get(theirs["id"], owner="session-a") is None
    assert store.delete(theirs["id"], owner="session-a") is False
    assert store.get(theirs["id"], owner="session-b") is not None
    # Without an owner (a shared, private install) everything is visible
    assert len(store.list()) == 2
    assert store.get(mine["id"]) is not None


def test_share_link_works_whoever_owns_the_analysis(store):
    saved = store.save("mine.pdf", "Data Analyst", result(), owner="session-a")

    assert store.get_shared(saved["share_token"])["id"] == saved["id"]


def test_database_from_before_share_links_is_upgraded(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE analyses (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, "
        "resume_filename TEXT NOT NULL, job_title TEXT NOT NULL, match_score REAL NOT NULL, "
        "ats_score REAL NOT NULL, result TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO analyses (created_at, resume_filename, job_title, match_score, ats_score, "
        "result) VALUES ('2026-10-01T00:00:00+00:00', 'old.pdf', 'Analyst', 50, 70, "
        """'{"match_score": 50, "ats": {"score": 70, "checks": []}}')"""
    )
    connection.commit()
    connection.close()

    store = HistoryStore(str(path))
    old = store.get(1)

    assert old["resume_filename"] == "old.pdf"
    assert old["share_token"]
    assert store.get_shared(old["share_token"])["id"] == 1
    assert store.save("new.pdf", "Analyst", result(), owner="s")["id"] == 2