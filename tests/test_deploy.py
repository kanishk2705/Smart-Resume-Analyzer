from deploy.huggingface.publish import CONTENTS, ROOT, assemble


def test_assembled_space_has_the_app_and_space_files(tmp_path):
    space = assemble(tmp_path / "space")

    for expected in [
        "Dockerfile",
        "README.md",
        "start.sh",
        ".streamlit/config.toml",
        "backend/main.py",
        "core/data/skills.json",
        "frontend/app.py",
        "requirements/backend.txt",
    ]:
        assert (space / expected).is_file(), expected
    assert (space / "README.md").read_text(encoding="utf-8").startswith("---\ntitle:")


def test_assembled_space_leaves_out_secrets_tests_and_caches(tmp_path):
    (ROOT / "core" / "__pycache__").mkdir(exist_ok=True)
    space = assemble(tmp_path / "space")

    names = {path.name for path in space.rglob("*")}

    assert ".env" not in names
    assert "__pycache__" not in names
    assert not (space / "tests").exists()
    assert not (space / "evaluation").exists()
    # The app's own Dockerfiles are for docker-compose; the Space has its own at the root
    assert not (space / "backend" / "Dockerfile").exists()


def test_start_script_has_unix_line_endings(tmp_path):
    space = assemble(tmp_path / "space")

    assert b"\r\n" not in (space / "start.sh").read_bytes()


def test_assembling_twice_replaces_the_previous_build(tmp_path):
    out = tmp_path / "space"
    assemble(out)
    (out / "stale.txt").write_text("left over")

    assemble(out)

    assert not (out / "stale.txt").exists()


def test_every_listed_source_exists():
    assert all((ROOT / source).exists() for source, _ in CONTENTS)
