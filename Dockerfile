FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download RKLLM runtime library (v1.2.3) and RKNN runtime library (v2.3.2)
RUN wget -q -O /usr/lib/librkllmrt.so \
    https://github.com/airockchip/rknn-llm/raw/refs/heads/master/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so \
 && wget -q -O /usr/lib/librknnrt.so \
    https://github.com/airockchip/rknn_model_zoo/raw/refs/heads/main/libs/Linux/aarch64/librknnrt.so \
 && ldconfig

# Install Python dependencies
# - flask + flask-cors + gunicorn: core server
# - numpy + Pillow: required for VL (vision-language) image processing
# - prometheus-flask-exporter + prometheus-client: optional metrics (graceful degradation if absent, but include for full feature set)
RUN pip install --no-cache-dir \
    flask \
    flask-cors \
    gunicorn \
    numpy \
    Pillow \
    prometheus-flask-exporter \
    prometheus-client

WORKDIR /app

COPY api.py gunicorn.config.py healthcheck.py ./

# Models are mounted at runtime — do not bake into image
VOLUME /root/models

EXPOSE 8000

ENV RKLLM_LOG_LEVEL=1 \
    RKLLM_API_LOG_LEVEL=INFO \
    GUNICORN_BIND=0.0.0.0:8000 \
    GUNICORN_WORKERS=1 \
    GUNICORN_THREADS=4 \
    GUNICORN_TIMEOUT=300

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python /app/healthcheck.py

CMD ["gunicorn", "--config", "gunicorn.config.py", "api:app"]
