FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config ./config
COPY src ./src
COPY scripts ./scripts
COPY models/haarcascade_frontalface_default.xml ./models/haarcascade_frontalface_default.xml
COPY models/reid/.gitkeep ./models/reid/.gitkeep
COPY deploy/entrypoint.sh /entrypoint.sh

RUN chmod 755 /entrypoint.sh \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/outputs /app/data /app/models/reid \
    && chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,sys; cmd=open('/proc/1/cmdline','rb').read(); sys.exit(0 if b'python' in cmd or b'xvfb' in cmd else 1)"

ENTRYPOINT ["/entrypoint.sh"]
