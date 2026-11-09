FROM python:3.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    HF_HOME=/model-cache \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt \
    && groupadd --gid 10001 rag \
    && useradd --uid 10001 --gid 10001 --create-home rag \
    && mkdir -p /app/data/processed /model-cache \
    && chown rag:rag /app/data/processed /model-cache

COPY app/ app/
COPY scripts/ scripts/
COPY evaluation/ evaluation/
COPY data/raw/ data/raw/
COPY data/sources.json data/LICENSE-kubernetes.txt data/

USER rag
RUN python scripts/build_corpus.py
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=9m --retries=3 \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=4).close()"]
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
