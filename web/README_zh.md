# Protein Solubility Predictor WebApp

这是一个基于 Django 的 Web 应用程序，用于预测蛋白质的溶解性、毒性、热稳定性等特性。它使用 Celery 进行异步任务处理，并集成了 ESM2 和 ESMC 等蛋白质大语言模型进行特征提取。

## 功能

- 用户可以输入蛋白质序列。
- 后端异步处理序列，提取特征并进行预测。
- 展示预测结果。

## 技术栈

- **后端:** Django, Django REST framework (可选, 如果有API)
- **异步任务:** Celery, Redis (作为 Broker 和 Backend)
- **特征提取:** ESM2, ESMC (通过本地模型或Hugging Face Hub)
- **数据库:** SQLite (默认, 可配置为 PostgreSQL 等)
- **部署:** Gunicorn (Python WSGI HTTP Server), Docker

## 本地开发环境运行指南

### 1. 环境准备

- Python 3.11+
- Conda (推荐用于环境管理)
- Redis Server
- (如果使用GPU) NVIDIA 显卡驱动 和 CUDA Toolkit 11.8+

### 2. 安装步骤

   a. **克隆项目 (如果从git获取):**
      ```bash
      git clone <your-repository-url>
      cd protein_webapp 
      ```

   b. **创建并激活 Conda 环境:**
      ```bash
      conda create -n protein_webapp_env python=3.11
      conda activate protein_webapp_env
      ```

   c. **安装依赖:**
      (确保 `requirements.txt` 文件在 `protein_webapp` 目录下并且是最新的)
      ```bash
      pip install -r requirements.txt
      ```
      *注意: `requirements.txt` 包含了 `torch` 的 CUDA 11.8 版本。如果你的 CUDA 版本不同或者你只想用 CPU，请相应修改此文件或单独安装 PyTorch。*

   d. **配置环境变量 (如果需要):**
      根据 `protein_webapp/settings.py` 可能需要配置一些环境变量，例如数据库连接、模型路径等。可以创建一个 `.env` 文件并使用 `python-dotenv` 加载，或者直接设置系统环境变量。
      例如，指定模型路径 (如果模型不由Hugging Face Hub自动下载且路径未硬编码的话):
      ```bash
      # .env 文件示例
      # ESM2_MODEL_PATH="/path/to/your/esm2_models/esm2_t33_650M_UR50D"
      # ESMC_MODEL_PATH="/path/to/your/esmc_models/esmc_600m"
      ```

   e. **运行数据库迁移:**
      ```bash
      python manage.py migrate
      ```

   f. **创建超级用户 (用于访问Django Admin):**
      ```bash
      python manage.py createsuperuser
      ```

### 3. 启动服务

   需要启动以下服务，建议在不同的终端窗口中分别启动：

   a. **启动 Redis Server:**
      (根据你的 Redis 安装方式启动，例如直接运行 `redis-server`)

   b. **启动 Celery Worker:**
      ```bash
      celery -A protein_webapp worker -l info
      ```

   c. **(可选) 启动 Celery Beat (如果定义了周期性任务):**
      ```bash
      celery -A protein_webapp beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler
      ```
      *(确保已安装 `django-celery-beat` 且在 `INSTALLED_APPS` 中配置并已迁移)*

   d. **启动 Django 开发服务器:**
      ```bash
      python manage.py runserver
      ```
      默认情况下，网站将运行在 `http://127.0.0.1:8000/`。

### 4. 重启服务 (本地开发)

在本地开发过程中若需重启服务 (例如，代码更改后需要完全重启生效)：

1.  **停止所有正在运行的服务：**
    *   在运行 Django 开发服务器 (`python manage.py runserver`) 的终端窗口中按 `Ctrl+C`。
    *   在运行 Celery Worker (`celery -A protein_webapp worker ...`) 的终端窗口中按 `Ctrl+C`。
    *   如果您正在运行 Celery Beat，也在其终端窗口中按 `Ctrl+C`。
    *   确保 Redis 服务器仍在运行，或者根据需要重启它 (通常应用程序代码更改不需要重启 Redis)。
2.  **重新启动服务**：按照上述 “3. 启动服务” 部分中的步骤操作。

## 使用 Docker 部署运行指南

本项目已配置 Docker 和 Docker Compose，可以方便地进行容器化部署。

### 1. 环境准备

- Docker Desktop (Windows, macOS) 或 Docker Engine (Linux)
- Docker Compose V2
- (如果使用GPU) NVIDIA 显卡驱动 和 NVIDIA Container Toolkit (Docker Desktop for Windows/Linux 通常会自动处理部分配置)

### 2. 配置文件说明

- `Dockerfile`: 定义了构建应用镜像的步骤，基于 `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` 以支持 GPU，并安装了 Python 3.11 和所有依赖。
- `docker-compose.yml`: 定义和编排了 `web` (Django + Gunicorn), `worker` (Celery worker), 和 `redis` 服务。配置为使用预构建的镜像 `cihebiyql/fusionprop:latest` 并支持 GPU。
- `requirements.txt`: Python 依赖列表。

### 3. 运行步骤

   a. **(首次或代码/依赖更新后) 构建镜像 (可选，如果使用预构建镜像则跳过):**
      如果你修改了 `Dockerfile` 或应用代码，并且不使用 `cihebiyql/fusionprop:latest`，则需要重新构建镜像。
      在 `protein_webapp` 目录下运行：
      ```bash
      docker-compose build
      ```

   b. **拉取预构建镜像 (推荐):**
      `docker-compose.yml` 已配置为使用 `cihebiyql/fusionprop:latest` 镜像。Docker Compose 会在启动时自动尝试拉取它。
      你也可以手动拉取：
      ```bash
      docker pull cihebiyql/fusionprop:latest
      ```

   c. **启动服务:**
      在 `protein_webapp` 目录下运行：
      ```bash
      docker-compose up -d
      ```
      (`-d` 参数表示在后台分离模式运行)。
      服务启动后，Django 应用将监听 `http://localhost:8000`。

   d. **(可选) 模型文件挂载:**
      - 当前 `docker-compose.yml` 未配置模型文件的显式卷挂载，假设模型会通过代码从 Hugging Face Hub 自动下载，或者模型路径已通过环境变量配置到容器内可访问的位置。
      - 如果你的模型文件较大且不希望每次都下载，或者有本地特定版本的模型，可以修改 `docker-compose.yml` 中的 `web` 和 `worker` 服务，添加 `volumes` 来挂载主机上的模型目录到容器内。
      例如，将主机上的 `./my_models_on_host` 目录挂载到容器内的 `/app/models`:
      ```yaml
      # docker-compose.yml (部分)
      services:
        web:
          image: cihebiyql/fusionprop:latest
          # ...
          volumes:
            - .:/app  # 应用代码
            - ./my_models_on_host:/app/models:ro # 挂载模型 (ro = read-only)
          environment:
            - ESM2_MODEL_PATH=/app/models/esm2_model_dir # 更新代码以使用此路径
            - ESMC_MODEL_PATH=/app/models/esmc_model_dir # 更新代码以使用此路径
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
      同时，确保你的 Python 代码 (例如 `FeatureManager` 或模型加载逻辑) 配置为从这些容器内路径 (`/app/models/...`) 读取模型，最好通过环境变量设置。

   e. **查看日志:**
      ```bash
      docker-compose logs -f # 查看所有服务的日志
      docker-compose logs -f web # 只查看 web 服务的日志
      docker-compose logs -f worker # 只查看 worker 服务的日志
      ```

   f. **停止服务:**
      ```bash
      docker-compose down # 停止并移除容器
      # docker-compose stop # 仅停止容器，不移除
      ```

   g. **重启服务 (Docker):**
      要在服务启动后重启 `docker-compose.yml` 中定义的所有服务：
      ```bash
      docker-compose restart
      ```
      此命令将尝试重启项目所有正在运行的容器。

      或者，如果您希望停止、移除然后重新创建容器（例如，为了应用镜像或配置的更改）：
      ```bash
      docker-compose down
      docker-compose up -d
      ```

### 4. 推送自定义镜像到 Docker Hub (开发者)

   如果你修改了代码并重新构建了本地镜像，想要推送到 Docker Hub (例如，推送到你自己的 `yourusername/yourimagename:latest`):
   ```bash
   # 1. 登录 Docker Hub
   docker login
   
   # 2. 找到本地构建的镜像名 (通常是 protein_webapp-web 或 protein_webapp_web)
   docker images
   
   # 3. 标记镜像
   docker tag <local_image_name> yourusername/yourimagename:latest
   
   # 4. 推送镜像
   docker push yourusername/yourimagename:latest
   ```
   之后，需要更新 `docker-compose.yml` 中的 `image:` 指令以使用新的镜像名。

## 注意事项

- **显存/内存管理:** 蛋白质大语言模型占用资源较多。确保你的开发和部署环境有足够的 RAM 和 GPU显存 (如果使用GPU)。我们已实现了模型在10分钟无任务后自动释放的机制。
- **模型路径配置:** 对于本地开发和 Docker 部署，确保模型路径被正确配置。推荐使用环境变量结合代码内的默认路径 (如 Hugging Face Hub ID) 来管理模型位置的灵活性。 