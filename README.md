# Smart Resume Analyzer

[![CI](https://github.com/kanishk2705/Smart-Resume-Analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/kanishk2705/Smart-Resume-Analyzer/actions/workflows/ci.yml)

Scores a resume against a job description using sentence embeddings, shows which resume
sentences drove the score, lists terms the resume is missing, and uses an LLM to suggest
how to close those gaps.

![Project Demo](assets/phase_4_demo.png)

## Why

Keyword-based applicant tracking systems reject candidates over wording: a resume that says
"built dashboards" can miss a job description asking for "data visualization". This project
compares the two documents by meaning as well as by keyword.

## Features

- **Semantic match score** — cosine similarity between SBERT (`all-MiniLM-L6-v2`) embeddings
  of the resume and the job description.
- **Hero sentences** — the three resume sentences most similar to the job description.
- **Missing keywords** — terms that appear in the job description but not in the resume,
  after NLTK cleaning and lemmatization.
- **AI learning path** — Google Gemini suggests a learning resource and a project idea for
  the top missing skills.
- **REST API** — everything the UI does is available over HTTP, with interactive docs.

## Architecture

```text
┌──────────────────┐   HTTP    ┌────────────────────┐   HTTPS   ┌────────────┐
│ Streamlit UI     │ ────────▶ │ FastAPI backend    │ ────────▶ │ Gemini API │
│ frontend/ :8501  │           │ backend/     :8000 │           └────────────┘
└──────────────────┘           │   └─ core/         │
                               │      parsing       │
                               │      cleaning      │
                               │      scoring       │
                               └────────────────────┘
```

The frontend holds no ML code and talks to the backend only over HTTP. `core/` has no web
framework code, so it can be tested and reused on its own.

```text
├── core/                  # Analysis logic, independent of any web framework
│   ├── config.py          # Settings from environment variables / .env
│   ├── parsing.py         # PDF and DOCX text extraction
│   ├── cleaning.py        # NLTK pipeline (stopwords, lemmatization)
│   ├── scoring.py         # Semantic score, hero sentences, keyword gap
│   └── recommendations.py # Gemini client with model fallback
├── backend/               # FastAPI service
│   ├── main.py            # Routes and error handling
│   ├── schemas.py         # Request / response models
│   └── Dockerfile
├── frontend/              # Streamlit client
│   ├── app.py
│   ├── api_client.py
│   └── Dockerfile
├── tests/                 # pytest suite
├── requirements/          # Pinned dependencies (backend, frontend, dev)
├── docker-compose.yml
└── .github/workflows/     # CI: lint + tests on every push
```

## Run with Docker

```bash
docker compose up --build
```

Then open <http://localhost:8501>. The API docs are at <http://localhost:8000/docs>.

The first build downloads PyTorch and the embedding model, so it takes a few minutes.

To enable the AI learning path, copy `.env.example` to `.env` and set `GEMINI_API_KEY`
before starting. Without it, the UI asks for a key when you use that feature.

## Run locally

Requires Python 3.11 or newer.

```bash
pip install -r requirements.txt

# Terminal 1: backend
uvicorn backend.main:app --reload

# Terminal 2: frontend
streamlit run frontend/app.py
```

The frontend looks for the backend at `http://localhost:8000`. Set the `BACKEND_URL`
environment variable to point it elsewhere.

## API

| Method | Path                      | Description                                           |
| ------ | ------------------------- | ----------------------------------------------------- |
| GET    | `/health`                 | Service status and configuration                      |
| POST   | `/api/v1/analyze`         | Score a resume (multipart upload) against a JD        |
| POST   | `/api/v1/recommendations` | Learning resources and projects for missing skills    |

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "resume=@resume.pdf" \
  -F "job_description=We are hiring a data analyst skilled in SQL and Python."
```

```json
{
  "match_score": 55.51,
  "missing_keywords": ["airflow", "aws", "cloud"],
  "top_sentences": [
    {"text": "Data analyst with four years of experience building dashboards.", "score": 0.63}
  ]
}
```

## Configuration

All settings are optional and read from environment variables or a `.env` file.

| Variable          | Default                      | Purpose                                      |
| ----------------- | ---------------------------- | -------------------------------------------- |
| `GEMINI_API_KEY`  | *(empty)*                    | Enables the AI learning path                 |
| `GEMINI_MODELS`   | see `.env.example`           | Models to try, in order                      |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2`           | Sentence-transformers model used for scoring |
| `MAX_UPLOAD_MB`   | `5`                          | Largest accepted resume file                 |
| `LOG_LEVEL`       | `INFO`                       | Backend log verbosity                        |
| `BACKEND_URL`     | `http://localhost:8000`      | Where the frontend finds the API             |

## Development

```bash
pip install -r requirements/dev.txt

ruff check .              # lint
pytest -m "not slow"      # fast tests, no model download
pytest                    # all tests, including one that loads the real model
```

## Known limitations

- The embedding model reads at most 256 tokens, so the overall score reflects only the
  first part of a long resume.
- The match score is a raw cosine similarity, not a calibrated probability.
- Text cleaning strips digits and symbols, so skills such as `C++` or `S3` are not
  matched correctly.
- Missing keywords are frequent word pairs from the job description, not verified skills.
- Scanned (image-only) PDFs are not supported.

## Roadmap

- [x] Phase 1: NLP keyword matching (NLTK)
- [x] Phase 2: Semantic matching and hero sentences (SBERT)
- [x] Phase 3: Docker
- [x] Phase 4: LLM learning paths (Gemini)
- [x] v2, step 1: FastAPI backend, separate frontend, tests, CI, pinned dependencies
- [ ] v2, step 2: Section-aware parsing, chunked scoring, score breakdown, skill extraction
- [ ] v2, step 3: Recruiter mode, ATS format checker, analysis history, PDF report

Earlier phases: [1](assets/phase_1_demo.png) · [2](assets/phase_2_demo.png) ·
[3](assets/phase_3_demo.png) · [4](assets/phase_4_demo.png)

---

Built by A C Kanishk.
