FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml ./
COPY src/ src/
RUN pip install ".[prod,kube]"
COPY examples/ examples/
COPY --from=web /web/dist web/dist
RUN useradd --uid 1000 --create-home forge && mkdir /data && chown forge /data
USER 1000
ENV FORGE_DATA_DIR=/data FORGE_WEB_DIR=/app/web/dist
EXPOSE 8080
# One worker process: the SSE fan-out and rate limiter are per process.
# Threads carry the long-lived stream connections.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--worker-class", "gthread", "--workers", "1", "--threads", "64", "--timeout", "0", "forge.api.wsgi:app"]
