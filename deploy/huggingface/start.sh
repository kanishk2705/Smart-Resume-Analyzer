#!/bin/sh
# Runs the API (reachable only inside the container) and the web UI (the public port).
set -e

uvicorn backend.main:app --host 127.0.0.1 --port 8000 &

cd frontend
# Spaces shows the app inside an iframe on huggingface.co, where Streamlit's XSRF check
# rejects file uploads; uploads are size-limited and parsed server-side either way.
exec streamlit run app.py \
    --server.port "${PORT:-7860}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableXsrfProtection false
