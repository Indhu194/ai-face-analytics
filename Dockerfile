# Multi-stage Python 3.11 base image for AI Face Analytics Platform
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV, ONNX Runtime, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8501 10000

# Start Streamlit application using shell form to bind to dynamic $PORT
CMD sh -c "streamlit run app/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"
