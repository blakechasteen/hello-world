# Multi-stage Dockerfile for Promptly Matrix Bot
# Production-ready with security hardening and optimization

# Stage 1: Base Python image with dependencies
FROM python:3.11-slim as base

LABEL maintainer="Promptly Team <hello@promptly.dev>"
LABEL description="Promptly Matrix Bot - Production deployment with full Phase 6 hardening"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user (non-root for security)
RUN useradd -m -u 1000 promptly && \
    mkdir -p /app /app/config /app/data /app/logs && \
    chown -R promptly:promptly /app

WORKDIR /app

# Stage 2: Dependencies installation
FROM base as dependencies

# Copy all requirements files
COPY requirements.txt requirements-github.txt requirements-production.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-github.txt && \
    pip install --no-cache-dir -r requirements-production.txt

# Stage 3: Application
FROM dependencies as application

# Copy application code
COPY bot/ ./bot/
COPY config/ ./config/
COPY --chown=promptly:promptly . .

# Switch to non-root user
USER promptly

# Expose ports
EXPOSE 8000 9090

# Health check (uses Phase 6B monitoring endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    PROMETHEUS_PORT=9090 \
    PYTHONPATH=/app

# Run application
CMD ["python", "-m", "bot.main"]
