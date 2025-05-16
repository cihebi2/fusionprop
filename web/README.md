# Protein Solubility Predictor WebApp

This is a Django-based web application for predicting protein properties such as solubility, toxicity, and thermal stability. It uses Celery for asynchronous task processing and integrates protein large language models like ESM2 and ESMC for feature extraction.

## Features

- Users can input protein sequences.
- The backend asynchronously processes sequences, extracts features, and makes predictions.
- Displays prediction results.

## Technology Stack

- **Backend:** Django, Django REST framework (optional, if APIs are present)
- **Asynchronous Tasks:** Celery, Redis (as Broker and Backend)
- **Feature Extraction:** ESM2, ESMC (via local models or Hugging Face Hub)
- **Database:** SQLite (default, configurable to PostgreSQL, etc.)
- **Deployment:** Gunicorn (Python WSGI HTTP Server), Docker

## Local Development Environment Setup Guide

### 1. Prerequisites

- Python 3.11+
- Conda (recommended for environment management)
- Redis Server
- (If using GPU) NVIDIA graphics card driver and CUDA Toolkit 11.8+

### 2. Installation Steps

   a. **Clone the project (if obtaining from git):**
      ```bash
      git clone <your-repository-url>
      cd protein_webapp 
      ```

   b. **Create and activate Conda environment:**
      ```bash
      conda create -n protein_webapp_env python=3.11
      conda activate protein_webapp_env
      ```

   c. **Install dependencies:**
      (Ensure the `requirements.txt` file is in the `protein_webapp` directory and is up-to-date)
      ```bash
      pip install -r requirements.txt
      ```
      *Note: `requirements.txt` includes the CUDA 11.8 version of `torch`. If you have a different CUDA version or only want to use CPU, please modify this file accordingly or install PyTorch separately.*

   d. **Configure environment variables (if needed):**
      Based on `protein_webapp/settings.py`, you might need to configure some environment variables, such as database connections, model paths, etc. You can create a `.env` file and use `python-dotenv` to load it, or set system environment variables directly.
      For example, to specify model paths (if models are not automatically downloaded from Hugging Face Hub and paths are not hardcoded):
      ```bash
      # .env file example
      # ESM2_MODEL_PATH="/path/to/your/esm2_models/esm2_t33_650M_UR50D"
      # ESMC_MODEL_PATH="/path/to/your/esmc_models/esmc_600m"
      ```

   e. **Run database migrations:**
      ```bash
      python manage.py migrate
      ```

   f. **Create a superuser (for accessing Django Admin):**
      ```bash
      python manage.py createsuperuser
      ```

### 3. Starting Services

   The following services need to be started, preferably in separate terminal windows:

   a. **Start Redis Server:**
      (Start according to your Redis installation method, e.g., by running `redis-server` directly)

   b. **Start Celery Worker:**
      ```bash
      celery -A protein_webapp worker -l info -P gevent
      ```

   c. **(Optional) Start Celery Beat (if periodic tasks are defined):**
      ```bash
      celery -A protein_webapp beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler
      ```
      *(Ensure `django-celery-beat` is installed, configured in `INSTALLED_APPS`, and migrations have been run)*

   d. **Start Django development server:**
      ```bash
      python manage.py runserver
      ```
      By default, the website will run at `http://127.0.0.1:8000/`.

### 4. Restarting Services (Local Development)

To restart the services during local development (e.g., after code changes that require a full restart):

1.  **Stop all running services:**
    *   Press `Ctrl+C` in the terminal window where the Django development server (`python manage.py runserver`) is running.
    *   Press `Ctrl+C` in the terminal window where the Celery Worker (`celery -A protein_webapp worker ...`) is running.
    *   If you are running Celery Beat, press `Ctrl+C` in its terminal window as well.
    *   Ensure the Redis server is still running or restart it if necessary (this typically doesn't need a restart for application code changes).
2.  **Restart services** by following the steps outlined in section "3. Starting Services" above.

## Docker Deployment Guide

This project is configured with Docker and Docker Compose for easy containerized deployment.

### 1. Prerequisites

- Docker Desktop (Windows, macOS) or Docker Engine (Linux)
- Docker Compose V2
- (If using GPU) NVIDIA graphics card driver and NVIDIA Container Toolkit (Docker Desktop for Windows/Linux usually handles part of this configuration automatically)

### 2. Configuration Files Overview

- `Dockerfile`: Defines the steps to build the application image, based on `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` for GPU support, and installs Python 3.11 and all dependencies.
- `docker-compose.yml`: Defines and orchestrates the `web` (Django + Gunicorn), `worker` (Celery worker), and `redis` services. Configured to use the pre-built image `cihebiyql/fusionprop:latest` and supports GPU.
- `requirements.txt`: List of Python dependencies.

### 3. Running Steps

   a. **(First time or after code/dependency updates) Build images (optional, skip if using the pre-built image):**
      If you modified the `Dockerfile` or application code and are not using `cihebiyql/fusionprop:latest`, you need to rebuild the images.
      In the `protein_webapp` directory, run:
      ```bash
      docker-compose build
      ```

   b. **Pull pre-built image (recommended):**
      `docker-compose.yml` is configured to use the `cihebiyql/fusionprop:latest` image. Docker Compose will automatically try to pull it on startup.
      You can also pull it manually:
      ```bash
      docker pull cihebiyql/fusionprop:latest
      ```

   c. **Start services:**
      In the `protein_webapp` directory, run:
      ```bash
      docker-compose up -d
      ```
      (The `-d` flag means run in detached mode in the background).
      After services start, the Django application will listen on `http://localhost:8000`.

   d. **(Optional) Mount model files:**
      - The current `docker-compose.yml` does not configure explicit volume mounts for model files. It assumes models will be automatically downloaded from Hugging Face Hub via code, or model paths are configured via environment variables to an accessible location within the container.
      - If your model files are large and you don't want to download them every time, or if you have local specific versions of models, you can modify the `web` and `worker` services in `docker-compose.yml` to add `volumes` for mounting model directories from the host to the container.
      For example, to mount the `./my_models_on_host` directory on the host to `/app/models` in the container:
      ```yaml
      # docker-compose.yml (partial)
      services:
        web:
          image: cihebiyql/fusionprop:latest
          # ...
          volumes:
            - .:/app  # Application code
            - ./my_models_on_host:/app/models:ro # Mount models (ro = read-only)
          environment:
            - ESM2_MODEL_PATH=/app/models/esm2_model_dir # Update code to use this path
            - ESMC_MODEL_PATH=/app/models/esmc_model_dir # Update code to use this path
        worker:
          image: cihebiyql/fusionprop:latest
          # ...
          volumes:
            - .:/app
            - ./my_models_on_host:/app/models:ro
          environment:
            - ESM2_MODEL_PATH=/app/models/esm2_model_dir
            - ESMC_MODEL_PATH=/app/models/esmc_model_dir
      ```
      Also, ensure your Python code (e.g., `FeatureManager` or model loading logic) is configured to read models from these container paths (`/app/models/...`), preferably set via environment variables.

   e. **View logs:**
      ```bash
      docker-compose logs -f # View logs for all services
      docker-compose logs -f web # View logs for web service only
      docker-compose logs -f worker # View logs for worker service only
      ```

   f. **Stop services:**
      ```bash
      docker-compose down # Stop and remove containers
      # docker-compose stop # Stop containers only, do not remove
      ```

   g. **Restarting Services (Docker):**
      To restart all services defined in `docker-compose.yml` after they have been started:
      ```bash
      docker-compose restart
      ```
      This command will attempt to restart all running containers for the project.

      Alternatively, if you want to stop, remove, and then recreate the containers (e.g., to apply changes in the image or configuration):
      ```bash
      docker-compose down
      docker-compose up -d
      ```

### 4. Pushing Custom Images to Docker Hub (Developer)

   If you have modified the code and rebuilt local images, and want to push them to Docker Hub (e.g., to your own `yourusername/yourimagename:latest`):
   ```bash
   # 1. Login to Docker Hub
   docker login
   
   # 2. Find the locally built image name (usually protein_webapp-web or protein_webapp_web)
   docker images
   
   # 3. Tag the image
   docker tag <local_image_name> yourusername/yourimagename:latest
   
   # 4. Push the image
   docker push yourusername/yourimagename:latest
   ```
   Afterward, you'll need to update the `image:` directive in `docker-compose.yml` to use the new image name.

## Important Notes

- **GPU/Memory Management:** Protein large language models consume significant resources. Ensure your development and deployment environments have sufficient RAM and GPU VRAM (if using GPU). We have implemented a mechanism for models to auto-release after 10 minutes of inactivity.
- **Model Path Configuration:** For both local development and Docker deployment, ensure model paths are correctly configured. It's recommended to use environment variables combined with default paths in the code (like Hugging Face Hub IDs) to manage model location flexibility.
- **Chinese Documentation:** A Chinese version of this document is available at [README_zh.md](README_zh.md).