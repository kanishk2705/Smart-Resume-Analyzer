# One-container build of the app for Hugging Face Spaces (or any host that runs a single
# container on one port). publish.py assembles the build context; see deploy/README.md.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/models/huggingface \
    NLTK_DATA=/opt/models/nltk_data \
    # Settings for a public demo: each visitor sees only their own history, which lives
    # on the container's temporary disk, and AI coach use is capped per visitor
    HISTORY_SCOPE=session \
    HISTORY_DB=/tmp/history.db \
    COACH_REQUESTS_PER_HOUR=20 \
    BACKEND_URL=http://127.0.0.1:8000

WORKDIR /app

# Install CPU Torch FIRST, so the pinned version in requirements doesn't pull the GPU build
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements/backend.txt -r requirements/frontend.txt

# Cache Warming (pre-download NLTK data and the embedding model into the image)
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/opt/models/nltk_data')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY core/ core/
COPY backend/ backend/
COPY frontend/ frontend/
COPY .streamlit/ frontend/.streamlit/
COPY start.sh .

# Spaces runs containers as user 1000
RUN useradd --create-home --uid 1000 user && chown -R user /opt/models
USER user

# The model is already in the image, so don't check the Hub for updates at startup
ENV HF_HUB_OFFLINE=1

EXPOSE 7860
CMD ["sh", "start.sh"]
