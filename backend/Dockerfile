# Build from the repository root: docker build -f backend/Dockerfile .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/models/huggingface \
    NLTK_DATA=/opt/models/nltk_data

WORKDIR /app

# Install CPU Torch FIRST, so the pinned version in requirements doesn't pull the GPU build
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements/backend.txt requirements/backend.txt
RUN pip install --no-cache-dir -r requirements/backend.txt

# Cache Warming (pre-download NLTK data and the embedding model into the image)
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/opt/models/nltk_data')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY core/ core/
COPY backend/ backend/

RUN useradd --create-home appuser && chown -R appuser /opt/models
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
