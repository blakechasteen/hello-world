# EdWIN AI Tutor - Production Dockerfile
# Multi-stage build for optimized container size
#
# Build: docker build -t edwin-ai-tutor:latest .
# Run: docker-compose up
#
# Implementation Date: November 15, 2025

# ============================================================================
# Stage 1: Builder - Install dependencies
# ============================================================================
FROM python:3.11-slim as builder

LABEL maintainer="EdWIN Team <edwin@eduverse.ai>"
LABEL description="EdWIN AI Tutor - K-12 Adaptive Learning Platform"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first (for layer caching)
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /tmp/requirements.txt && \
    pip install \
    fastapi \
    uvicorn[standard] \
    python-jose[cryptography] \
    passlib[bcrypt] \
    python-multipart \
    redis \
    neo4j \
    qdrant-client

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    EDWIN_ENV=production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create app user (non-root for security)
RUN useradd -m -u 1000 edwin && \
    mkdir -p /app && \
    chown -R edwin:edwin /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=edwin:edwin . /app/

# Switch to non-root user
USER edwin

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "EduVerse.edwin.api:app", "--host", "0.0.0.0", "--port", "8000"]
