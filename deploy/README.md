# Deploying a public demo

`docker compose up` runs the app as two containers for local use. A public demo needs the
settings a public site needs, which both deploy options below switch on:

| Setting                   | Value     | Effect                                                                     |
| ------------------------- | --------- | -------------------------------------------------------------------------- |
| `HISTORY_SCOPE`           | `session` | Each visitor sees only their own history, for as long as the tab is open   |
| `HISTORY_DB`              | temp dir  | History lives on the host's temporary disk and is wiped on restart         |
| `COACH_REQUESTS_PER_HOUR` | `20`      | Caps AI coach use per visitor, so one visitor cannot exhaust the Groq quota |

## Streamlit Community Cloud (free)

Community Cloud runs a single Streamlit process, so
[streamlit/streamlit_app.py](streamlit/streamlit_app.py) runs the API inside it
(`BACKEND_URL=inprocess`) and [streamlit/requirements.txt](streamlit/requirements.txt)
installs PyTorch's CPU-only build, which is a fraction of the size of the default one.

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **Create app → Deploy a public app from GitHub**, and fill in:
   - Repository: `kanishk2705/Smart-Resume-Analyzer`
   - Branch: `main`
   - Main file path: `deploy/streamlit/streamlit_app.py`
   - App URL: a subdomain of your choice, for example `smart-resume-analyzer`
3. Open **Advanced settings**, choose **Python 3.12**, and paste into **Secrets**:

   ```toml
   GROQ_API_KEY = "gsk_..."
   APP_URL = "https://smart-resume-analyzer.streamlit.app"
   ```

   `APP_URL` must match the App URL from step 2; share links are built from it. Leave
   out `GROQ_API_KEY` to run without the AI coach.
4. **Deploy.** The first build takes several minutes. The first analysis after a cold start
   is slower, because the embedding model is downloaded then.

Free apps go to sleep after a period without visitors; the next visitor wakes them up.

## Hugging Face Spaces (paid)

Hugging Face now requires a PRO subscription to run Docker Spaces. With one,
`deploy/huggingface/` holds a single-container build and a script that uploads it.

Put a write-access token in `.env` as `HF_TOKEN=hf_...`, then:

```bash
pip install -r requirements/eval.txt   # includes huggingface_hub
python -m deploy.huggingface.publish --space <your-username>/smart-resume-analyzer --groq-key-from-env
```

This creates the Space, stores `GROQ_API_KEY` as a Space **secret**, sets `APP_URL`, and
uploads the app.

To try the single-container build locally:

```bash
python -m deploy.huggingface.publish --out build/space
docker build -t resume-analyzer-space build/space
docker run -p 7860:7860 -e GROQ_API_KEY resume-analyzer-space
```
