"""Builds the Hugging Face Space from this repository and, optionally, uploads it.

    python -m deploy.huggingface.publish --out build/space            # assemble only
    python -m deploy.huggingface.publish --space <user>/<name>        # assemble and upload

Uploading needs a Hugging Face access token with write permission, in the HF_TOKEN
environment variable or .env. With --groq-key-from-env, the GROQ_API_KEY from the
environment (or .env) is stored as a Space secret, so it never appears in the uploaded files.
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# What the Space needs, as (source in this repository, path in the Space)
CONTENTS = [
    ("core", "core"),
    ("backend", "backend"),
    ("frontend", "frontend"),
    ("requirements/backend.txt", "requirements/backend.txt"),
    ("requirements/frontend.txt", "requirements/frontend.txt"),
    (".streamlit", ".streamlit"),
    ("deploy/huggingface/Dockerfile", "Dockerfile"),
    ("deploy/huggingface/start.sh", "start.sh"),
    ("deploy/huggingface/SPACE_README.md", "README.md"),
]
IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", "Dockerfile", ".env")


def assemble(out: Path) -> Path:
    """Copies the Space's files into `out`, replacing anything already there."""
    if out.exists():
        shutil.rmtree(out)
    for source, target in CONTENTS:
        source_path, target_path = ROOT / source, out / target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_dir():
            shutil.copytree(source_path, target_path, ignore=IGNORED)
        else:
            shutil.copy2(source_path, target_path)
    # start.sh must keep Unix line endings to run in the container
    start = out / "start.sh"
    start.write_bytes(start.read_bytes().replace(b"\r\n", b"\n"))
    return out


def secret_from_env(name: str) -> str:
    """A secret from the environment, or failing that from the repository's .env file."""
    value = os.getenv(name, "")
    env_file = ROOT / ".env"
    if not value and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{name}="):
                value = line.split("=", 1)[1].strip()
    if not value:
        raise SystemExit(f"{name} is not set in the environment or .env")
    return value


def upload(folder: Path, space: str, groq_key: str | None) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=secret_from_env("HF_TOKEN"))
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    owner, name = space.split("/")
    url = f"https://{owner}-{name}".lower().replace("_", "-") + ".hf.space"
    # Share links point at the Space's own address
    api.add_space_variable(space, "APP_URL", url)
    if groq_key:
        api.add_space_secret(space, "GROQ_API_KEY", groq_key)
    api.upload_folder(
        repo_id=space,
        repo_type="space",
        folder_path=folder,
        commit_message="Deploy from github.com/kanishk2705/Smart-Resume-Analyzer",
        delete_patterns=["*"],  # remove files that are no longer part of the app
    )
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, help="Where to assemble the Space (default: temporary)")
    parser.add_argument("--space", help="Upload to this Space, as <user>/<name>")
    parser.add_argument("--groq-key-from-env", action="store_true")
    args = parser.parse_args()

    folder = assemble(args.out or Path(tempfile.mkdtemp()) / "space")
    print(f"Assembled the Space in {folder}")
    if args.space:
        key = secret_from_env("GROQ_API_KEY") if args.groq_key_from_env else None
        url = upload(folder, args.space, key)
        print(f"Uploaded. The Space builds itself; it will be live at {url} in a few minutes.")


if __name__ == "__main__":
    main()
