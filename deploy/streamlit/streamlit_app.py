"""Entrypoint for Streamlit Community Cloud, which runs one Streamlit process and nothing else.

The API runs inside the same process (BACKEND_URL=inprocess), with the settings a public
demo needs. Secrets set in the app's settings (GROQ_API_KEY, APP_URL) arrive as
environment variables and take precedence over the defaults below.
"""

import os
import runpy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PUBLIC_DEMO_SETTINGS = {
    "BACKEND_URL": "inprocess",
    # Each visitor sees only their own history, kept on the host's temporary disk
    "HISTORY_SCOPE": "session",
    "HISTORY_DB": str(Path(tempfile.gettempdir()) / "resume-analyzer-history.db"),
    "COACH_REQUESTS_PER_HOUR": "20",
}
for name, value in PUBLIC_DEMO_SETTINGS.items():
    os.environ.setdefault(name, value)

for path in (ROOT / "frontend", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

runpy.run_path(str(ROOT / "frontend" / "app.py"), run_name="__main__")
