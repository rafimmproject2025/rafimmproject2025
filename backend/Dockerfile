# Backend Dockerfile (robust)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System deps often needed for scientific/DB libs (hdbscan, numpy/scipy, psycopg from source, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      gcc g++ \
      git \
      curl \
      libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip/setuptools/wheel first
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel

# If you use sentence-transformers (pulls torch), pre-install *CPU* wheels to avoid CUDA / index issues.
# (Safe to keep; it no-ops if not needed. Remove if your requirements already pin torch properly.)
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      'torch>=2.2,<2.4' 'torchvision>=0.17,<0.19' 'torchaudio>=2.2,<2.4' || true

# Now your project deps
RUN python -m pip install --no-cache-dir -r requirements.txt



# Pre-cache sentence-transformers model at build time
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("cached all-MiniLM-L6-v2")
PY

# # Optional: run offline at runtime
# ENV TRANSFORMERS_OFFLINE=1
# ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/sentence-transformers

# Copy source after deps to leverage Docker layer cache
COPY . .

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]


# FROM python:3.11-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . .
# ENV PYTHONUNBUFFERED=1

# # sentence-transformers model name (used by cosine + update pipeline)
# ENV EMB_MODEL=all-MiniLM-L6-v2
# # <-- This is the important line for your question:
# ENV AUTO_PIPELINE_SCRIPT=/app/auto_category_pipeline.py

# CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
