# Use official CUDA base image for PyTorch with CUDA 11.8
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python

# Upgrade pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . /app/

# Expose port 8000 for Django app
EXPOSE 8000

# Default command (overridden by docker-compose)
# CMD ["gunicorn", "protein_webapp.wsgi:application", "--bind", "0.0.0.0:8000"] 