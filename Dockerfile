# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Install uv (dependency and virtualenv manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install system dependencies required for scientific computing
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching.
# Copy only the lockfiles so this layer is cached unless deps change.
ENV UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Create models directory if it doesn't exist
RUN mkdir -p models

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the application
ENTRYPOINT ["uv", "run", "--no-dev", "streamlit", "run", "NTPN_APP.py", "--server.port=8501", "--server.address=0.0.0.0"]
