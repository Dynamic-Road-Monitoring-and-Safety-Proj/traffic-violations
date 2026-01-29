# =============================================================================
# Traffic Violation Detection System - Dockerfile
# =============================================================================
# Multi-stage build for production deployment
# Optimized for AWS ECS/EC2 deployment with optional GPU support

# -----------------------------------------------------------------------------
# Stage 1: Base image with Python and system dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -m appuser

# Set working directory
WORKDIR /app

# -----------------------------------------------------------------------------
# Stage 2: Builder - Install Python dependencies
# -----------------------------------------------------------------------------
FROM base as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 3: Production image
# -----------------------------------------------------------------------------
FROM base as production

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser . /app/

# Create necessary directories
RUN mkdir -p /app/output /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port for potential API server (future use)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import traffic_violation_detector; print('OK')" || exit 1

# Default command - can be overridden
ENTRYPOINT ["python", "-m", "traffic_violation_detector"]
CMD ["--help"]

# -----------------------------------------------------------------------------
# Stage 4: GPU-enabled image (for NVIDIA GPU support)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04 as gpu

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Create non-root user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -m appuser

WORKDIR /app

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install onnxruntime-gpu

# Copy application code
COPY --chown=appuser:appuser . /app/

# Create directories
RUN mkdir -p /app/output /app/logs && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import traffic_violation_detector; print('OK')" || exit 1

ENTRYPOINT ["python", "-m", "traffic_violation_detector"]
CMD ["--help"]
