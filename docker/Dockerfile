FROM rust:1.86-bookworm AS tectonic-builder

RUN cargo install --locked tectonic

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib-cache \
    XDG_CACHE_HOME=/tmp/xdg-cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=tectonic-builder /usr/local/cargo/bin/tectonic /usr/local/bin/tectonic

COPY requirements.txt /app/requirements.txt
COPY src/backend/requirements.txt /app/src/backend/requirements.txt

RUN python -m pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app

RUN chmod +x /app/docker/entrypoint.sh

ENTRYPOINT ["/app/docker/entrypoint.sh"]
