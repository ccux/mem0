# ✅ Phase 1.1C.1: Multi-Stage Build Implementation
# Optimized Dockerfile with multi-stage build for better performance and smaller image size

# =============================================================================
# Stage 1: Build Dependencies (Builder Stage)
# =============================================================================
FROM python:3.12-slim as builder

# Set environment variables for build optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for better dependency isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory for building
WORKDIR /app/build

# Copy and install mem0 library dependencies first (better caching)
COPY pyproject.toml poetry.lock* README.md LICENSE ./
COPY ./mem0 ./mem0/

# Install the local mem0 library in the virtual environment
RUN pip install --no-cache-dir -e .

# Copy server requirements and install them
COPY server/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Production Runtime (Final Stage)
# =============================================================================
FROM python:3.12-slim as production

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Doppler CLI (production secret management)
RUN curl -Ls https://cli.doppler.com/install.sh | sh

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create app user for security (non-root)
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy wait-for-it.sh script and make it executable
COPY server/wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

# Copy application code (only what's needed for runtime)
COPY server/main.py ./
COPY server/.env.example ./

# Change ownership to app user
RUN chown -R appuser:appuser /app

# Switch to non-root user for security
USER appuser

# Expose port
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command (no reload for better performance)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
