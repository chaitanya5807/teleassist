FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf-cache

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install -r requirements.txt && pip install fastapi uvicorn streamlit requests

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install -e . --no-deps

# Retrieval index (built locally, not stored in git)
COPY data/processed/chunks_official.jsonl data/processed/dense_index_official.npz ./data/processed/

EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "teleassist.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
