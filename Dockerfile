FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

WORKDIR /app

# libgomp1 is needed by libraries such as FAISS
# and some scientific Python packages.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies separately so Docker can
# reuse this layer when application code changes.
COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# Create an unprivileged application user.
RUN groupadd --system chatbot \
    && useradd \
        --system \
        --gid chatbot \
        --create-home \
        chatbot

COPY --chown=chatbot:chatbot . .

# Ensure the runtime log directory is writable.
RUN mkdir -p /app/data /app/logs \
    && chown -R chatbot:chatbot /app/data /app/logs

USER chatbot

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=120s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=5)" || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]