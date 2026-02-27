# hololoom SaaS API Dockerfile
#
# Build:
#   docker build -t hololoom-saas -f hololoom/saas/Dockerfile .
#
# Run (SQLite, development):
#   docker run -p 8000:8000 hololoom-saas
#
# Run (PostgreSQL, production):
#   docker run -p 8000:8000 \
#     -e DATABASE_URL=postgresql://user:pass@host:5432/db \
#     -e ENABLE_BILLING=true \
#     -e STRIPE_API_KEY=sk_... \
#     hololoom-saas

# ============================================================================
# Build stage
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user \
    fastapi \
    uvicorn[standard] \
    pydantic \
    httpx \
    aiohttp \
    asyncpg \
    aiosqlite \
    bcrypt \
    python-multipart

# ============================================================================
# Runtime stage
# ============================================================================
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY hololoom /app/hololoom

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default configuration
ENV PORT=8000
ENV HOST=0.0.0.0
ENV WORKERS=4
ENV LOG_LEVEL=INFO
ENV ENABLE_BILLING=false
ENV ENABLE_USAGE_TRACKING=true
ENV ENABLE_AUDIT_LOG=true

# Create data directory for SQLite
RUN mkdir -p /app/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD wget --spider http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Run the application
CMD ["sh", "-c", "uvicorn hololoom.saas.app:app --host $HOST --port $PORT --workers $WORKERS"]
