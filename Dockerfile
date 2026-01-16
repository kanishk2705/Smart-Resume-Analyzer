# 1. Base Image
FROM python:3.10-slim

# 2. Optimization
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Work Directory
WORKDIR /app

# 4. "THE BODYGUARD MOVE": Install CPU Torch FIRST
# We do this BEFORE copying requirements so it's cached and blocks the GPU version
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. Install the rest of the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Code
COPY . .

# 7. Cache Warming (Pre-download AI models)
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 8. Run
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]