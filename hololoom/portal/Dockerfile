# Portal WASM Execution System - Multi-Stage Dockerfile
#
# This Dockerfile builds both Portal Server and Node Daemon components
# using multi-stage builds to optimize image size and build time.
#
# Build targets:
#   - portal-server: Control plane for node management
#   - node-daemon: WASM job execution worker
#
# Usage:
#   docker build --target portal-server -t portal-server:latest .
#   docker build --target node-daemon -t node-daemon:latest .
#
# Or use docker-compose which handles targets automatically.

# =============================================================================
# Stage 1: Base Builder - Install dependencies
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-portal.txt .
RUN pip install --no-cache-dir --user -r requirements-portal.txt

# =============================================================================
# Stage 2: Portal Server Runtime
# =============================================================================
FROM python:3.11-slim as portal-server

LABEL maintainer="hololoom Team"
LABEL description="Portal Server - Control plane for distributed WASM execution"
LABEL version="0.1.0"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy Portal shared library and server code
COPY hololoom/portal/shared /app/hololoom/portal/shared
COPY hololoom/portal/portal_server /app/hololoom/portal/portal_server
COPY hololoom/portal/__init__.py /app/hololoom/portal/__init__.py

# Create empty hololoom package
RUN mkdir -p /app/hololoom && touch /app/hololoom/__init__.py

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default port
EXPOSE 8080

# Run Portal Server
CMD ["python", "-m", "hololoom.portal.portal_server.main"]

# =============================================================================
# Stage 3: Node Daemon Runtime
# =============================================================================
FROM python:3.11-slim as node-daemon

LABEL maintainer="hololoom Team"
LABEL description="Node Daemon - WASM job execution worker"
LABEL version="0.1.0"

WORKDIR /app

# Install runtime dependencies including WASM runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy Portal shared library and daemon code
COPY hololoom/portal/shared /app/hololoom/portal/shared
COPY hololoom/portal/node_daemon /app/hololoom/portal/node_daemon
COPY hololoom/portal/__init__.py /app/hololoom/portal/__init__.py

# Create empty hololoom package
RUN mkdir -p /app/hololoom && touch /app/hololoom/__init__.py

# Create WASM modules directory
RUN mkdir -p /app/wasm_modules

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV NODE_WASM_DIR=/app/wasm_modules

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1

# Default port
EXPOSE 9090

# Run Node Daemon
CMD ["python", "-m", "hololoom.portal.node_daemon.main"]
