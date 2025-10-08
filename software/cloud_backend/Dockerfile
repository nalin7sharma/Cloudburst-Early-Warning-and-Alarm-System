# Cloudburst Backend Dockerfile
# Multi-stage build for optimized production image

# Stage 1: Builder
FROM python:3.9-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY software/cloud_backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim as runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=5000
ENV ENVIRONMENT=production

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r cloudburst && useradd -r -g cloudburst cloudburst

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/cloudburst/.local
ENV PATH=/home/cloudburst/.local/bin:$PATH

# Copy application code
COPY software/cloud_backend/ ./cloud_backend/
COPY firmware/ ./firmware/
COPY docs/ ./docs/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models && \
    chown -R cloudburst:cloudburst /app

# Switch to non-root user
USER cloudburst

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/system/health || exit 1

# Start application
CMD ["gunicorn", \
    "--bind", "0.0.0.0:5000", \
    "--workers", "4", \
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--access-logfile", "-", \
    "--error-logfile", "-", \
    "--timeout", "120", \
    "--preload", \
    "cloud_backend.app:app"]
