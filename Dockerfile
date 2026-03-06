FROM python:3.11-slim

WORKDIR /app

ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache

COPY requirements.txt .
RUN pip install --upgrade pip

# Install CPU-only torch first (200MB instead of 915MB)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install rest of requirements
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

RUN mkdir -p docs faiss_index mlruns static

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT