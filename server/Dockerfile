# Ultra-minimal Mem0 Dockerfile to prevent ENOSPC errors
# Optimized multi-stage build for minimal disk usage

# =============================================================================
# Stage 1: Build Dependencies (Ultra-minimal Builder)
# =============================================================================
FROM python:3.12-slim as builder

# Set environment variables for build optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_WARN_SCRIPT_LOCATION=1

# Install minimal build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create ultra-minimal virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory for building
WORKDIR /app/build

# Copy and install dependencies with aggressive optimization
COPY pyproject.toml poetry.lock* README.md LICENSE ./
COPY ./mem0 ./mem0/

# Install mem0 library with minimal footprint
RUN pip install --no-cache-dir --no-deps -e . && \
    # Remove build artifacts immediately
    find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Install server requirements with cleanup
COPY server/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    # Aggressive cleanup of build artifacts
    find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -name "*.pyo" -delete 2>/dev/null || true && \
    find /opt/venv -name "*.pyd" -delete 2>/dev/null || true && \
    find /opt/venv -name "*.so" -exec strip {} \; 2>/dev/null || true && \
    # Remove test files and documentation
    find /opt/venv -name "test*" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -name "*test*" -type f -delete 2>/dev/null || true && \
    find /opt/venv -name "*.md" -delete 2>/dev/null || true && \
    find /opt/venv -name "*.txt" -delete 2>/dev/null || true && \
    find /opt/venv -name "LICENSE*" -delete 2>/dev/null || true && \
    find /opt/venv -name "README*" -delete 2>/dev/null || true

# =============================================================================
# Stage 2: Ultra-minimal Production Runtime
# =============================================================================
FROM python:3.12-slim as production

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Install only essential runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*

# Install Doppler CLI for production secret management
RUN curl -Ls https://cli.doppler.com/install.sh | sh && \
    rm -rf /tmp/* /var/tmp/*

# Copy optimized virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy minimal application files
COPY server/wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

COPY server/main.py ./
COPY server/.env.example ./

# Set ownership and clean up
RUN chown -R appuser:appuser /app && \
    rm -rf /tmp/* /var/tmp/*

# Use non-root user for security
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Labels
LABEL maintainer="Cognition Suite Team"
LABEL version="1.0.0"
LABEL description="Mem0 AI Service - Ultra-minimal Production"
