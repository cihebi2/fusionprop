# FusionProp: 蛋白质特性预测平台

FusionProp 是一个基于 Web 的平台，旨在快速、准确地预测蛋白质的多种关键特性，包括溶解度、热稳定性和毒性。它利用深度学习模型和先进的蛋白质语言模型 (PLM) 特征，直接从氨基酸序列进行预测，无需蛋白质三维结构。

## 主要特性

*   **多任务预测:** 一次运行即可同时获取溶解度、热稳定性和毒性的预测结果。
*   **无需结构:** 完全基于序列进行预测，避免了耗时且计算密集的蛋白质结构预测步骤。
*   **高性能:** 采用优化的模型和异步任务处理，实现快速预测，适合高通量筛选。
*   **高精度:** 模型在基准数据集上达到或接近当前最佳性能。
*   **用户友好:** 提供简洁的 Web 界面，支持直接粘贴序列或上传 FASTA 文件，结果清晰展示并可下载。
*   **模块化设计:** 包含用于特征提取、模型训练和 Web 应用服务的独立模块。

## 技术栈

*   **后端:** Django
*   **异步任务:** Celery, Redis (作为消息代理和结果后端)
*   **机器学习/深度学习:** PyTorch
*   **蛋白质语言模型:**
    *   ESM (例如 Facebook 的 ESM-2 系列)
    *   ESMC
*   **特征提取:** 使用 `transformers` 库 (针对 ESM-2) 和 `esm` (OpenFold 团队的 ESM) (针对 ESMC)
*   **数据库:** SQLite (默认, 可配置为 PostgreSQL 等)
*   **Web 服务器/部署:** Gunicorn, Docker, Nginx (推荐用于生产环境反向代理)
*   **前端:** HTML, CSS, JavaScript, Bootstrap

## 目录结构

```
fusionprop/
├── data/                     # 存储训练、测试数据集及相关原始数据
├── extract_features/         # 包含使用不同PLM提取蛋白质特征的脚本
│   ├── extract_esm_1.py      # 使用ESM-2提取特征
│   ├── extract_esmc_1.py     # 使用ESMC提取特征
│   └── ...                   # 其他特征提取脚本和相关shell脚本
├── train_script/             # 包含训练不同特性预测模型的脚本
│   ├── solubility/           # 溶解度模型训练脚本 (e.g., fusion_5_5_4_2_3.py)
│   ├── thermostability/      # 热稳定性模型训练脚本 (e.g., train_22_1_1.py)
│   └── toxicity/             # 毒性模型训练脚本 (e.g., train_12_2.py, evaluate_model.py)
├── web/                      # Django Web应用和API的核心代码 (原项目的主体)
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── manage.py
│   ├── predictor/            # Django app，处理预测逻辑、表单、视图、任务
│   ├── protein_feature_extractor/ # 特征提取器管理模块
│   ├── protein_predictor/    # 各个预测模型的管理和实现
│   ├── protein_webapp/       # Django项目配置 (settings, urls, celery)
│   ├── requirements.txt
│   ├── static/
│   └── templates/
├── .gitattributes            # Git LFS 跟踪规则
├── README.md                 # 本文件
├── README_zh.md              # 中文版 README
└── web_environment.yml       # Conda 环境依赖文件 (由此项目生成)
```

## 环境设置与安装

### 1. 先决条件

*   Python 3.11+
*   Conda (推荐用于环境管理)
*   Redis 服务器
*   (可选, 若使用GPU) NVIDIA 显卡驱动和 CUDA Toolkit (例如 11.8+)
*   Git LFS (用于处理大型数据和模型文件)

### 2. 安装步骤

a.  **克隆仓库:**
    ```bash
    git clone https://github.com/cihebi2/fusionprop.git
    cd fusionprop
    ```

b.  **安装 Git LFS:** (如果尚未安装)
    请参照 [Git LFS 官网](https://git-lfs.github.com/) 指南进行安装。然后在仓库内初始化：
    ```bash
    git lfs install
    git lfs pull # 拉取 LFS 管理的大文件
    ```

c.  **创建并激活 Conda 环境:**
    您可以使用提供的 `web_environment.yml` 文件来创建环境 (推荐):
    ```bash
    conda env create -f web_environment.yml
    conda activate web # 或 yml 文件中指定的环境名
    ```
    或者，如果您想手动创建 (类似原始 README 中的 protein_webapp_env):
    ```bash
    conda create -n fusionprop_env python=3.11
    conda activate fusionprop_env
    # 然后根据 web/requirements.txt 安装依赖 (可能需要调整以匹配 yml)
    # pip install -r web/requirements.txt
    ```

d.  **配置环境变量 (如果需要):**
    根据 `web/protein_webapp/settings.py`，您可能需要配置数据库连接、模型路径等。可以创建 `.env` 文件并使用 `python-dotenv` 加载，或直接设置系统环境变量。

e.  **数据库迁移 (针对Web应用):**
    ```bash
    cd web
    python manage.py migrate
    cd ..
    ```

f.  **创建超级用户 (可选, 用于访问Django Admin):**
    ```bash
    cd web
    python manage.py createsuperuser
    cd ..
    ```

## 运行 FusionProp

### 本地开发模式 (Web 应用)

确保 Conda 环境已激活且 Redis 服务器正在运行。

1.  **启动 Redis 服务器:**
    (根据您的 Redis 安装方式启动, 例如直接运行 `redis-server`)

2.  **启动 Celery Worker (在 `fusionprop/web/` 目录下):**
    ```bash
    cd web
    celery -A protein_webapp worker -l info -P gevent
    ```
    (根据需要，可以在另一个终端启动 Celery Beat: `celery -A protein_webapp beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler`)

3.  **启动 Django 开发服务器 (在 `fusionprop/web/` 目录下):**
    ```bash
    python manage.py runserver
    ```
    默认情况下，网站将运行在 `http://127.0.0.1:8000/`。

### Docker 部署模式

项目已配置 Docker 和 Docker Compose，方便容器化部署。

1.  **先决条件:**
    *   Docker Desktop (Windows, macOS) 或 Docker Engine (Linux)
    *   Docker Compose V2
    *   (可选, 若使用GPU) NVIDIA 显卡驱动和 NVIDIA Container Toolkit

2.  **配置文件:**
    *   `web/Dockerfile`: 定义应用镜像的构建步骤。
    *   `web/docker-compose.yml`: 定义和编排 `web` (Django + Gunicorn), `worker` (Celery worker), 和 `redis` 服务。

3.  **运行步骤 (在 `fusionprop/web/` 目录下):**
    a.  **构建镜像 (如果修改了 Dockerfile 或代码，并且不使用预构建镜像):**
        ```bash
        cd web
        docker-compose build
        ```
    b.  **启动服务:**
        ```bash
        docker-compose up -d
        ```
        ( `-d` 参数表示在后台分离模式运行)。服务启动后，Django 应用将监听 `http://localhost:8000` (或 docker-compose.yml 中配置的端口)。

    c.  **查看日志:**
        ```bash
        docker-compose logs -f
        docker-compose logs -f web
        docker-compose logs -f worker
        ```

    d.  **停止服务:**
        ```bash
        docker-compose down # 停止并移除容器
        # docker-compose stop # 仅停止容器，不移除
        ```
    e.  **重启服务:**
        ```bash
        docker-compose restart
        # 或
        # docker-compose down
        # docker-compose up -d
        ```

## 特征提取

`extract_features/` 目录包含用于从蛋白质序列中提取嵌入特征的脚本。这些特征随后可用于训练预测模型。

*   **`extract_esm_1.py`**: 使用 ESM-2 模型 (例如 `facebook/esm2_t33_650M_UR50D`) 提取特征。它会处理输入 CSV 文件，为每个序列生成残基级嵌入和平均池化的蛋白质整体表示，并将结果保存为 `.npy` 文件。脚本中包含序列填充和掩码逻辑。
*   **`extract_esmc_1.py`**: 使用 ESMC 模型 (例如 `esmc_600m`) 提取特征。与 ESM-2 脚本类似，但使用 CLS token 作为蛋白质的整体表示。
*   通常会附带 `.sh` 脚本来方便地运行这些 Python 脚本，并可能包含对输入文件和输出目录的参数化设置。

**运行示例 (概念性):**
```bash
cd extract_features
# conda activate <your_env_with_dependencies_like_transformers_esm>
# python extract_esm_1.py --input_csv ../data/your_sequences.csv --output_dir ./esm2_embeddings --model_name facebook/esm2_t33_650M_UR50D
# sh extract_esm_1.sh # (如果shell脚本已配置好参数)
cd ..
```
请根据脚本内的具体实现和 `if __name__ == "__main__":` 部分调整参数和路径。

## 模型训练

`train_script/` 目录包含用于训练不同蛋白质特性预测模型的脚本。每个子目录对应一种特性。

*   **`train_script/solubility/`**: 例如 `fusion_5_5_4_2_3.py` 和对应的 `.sh` 脚本，用于训练溶解度预测模型。这些脚本通常会加载预先提取的特征，定义模型架构 (如加权融合策略)，并执行训练和评估流程。
*   **`train_script/thermostability/`**: 例如 `train_22_1_1.py`，用于训练热稳定性预测模型。
*   **`train_script/toxicity/`**: 例如 `train_12_2.py` (训练) 和 `evaluate_model.py` (评估)，用于毒性预测模型。

**运行示例 (概念性):**
```bash
cd train_script/toxicity
# conda activate <your_env_with_training_dependencies_like_pytorch_pandas_sklearn>
# python train_12_2.py --feature_path ../../extract_features/esm2_embeddings/ --label_file ../../data/toxicity_labels.csv --save_path ./trained_toxicity_model/
# sh train_12_3_3.sh # (如果shell脚本已配置好参数)
cd ../..
```
具体的运行命令和所需参数请参照各个训练脚本内部的说明或其对应的 shell 脚本。

## 使用 Web 应用

当 Web 应用通过本地开发模式或 Docker 成功运行后：

1.  打开浏览器并访问 `http://localhost:8000` (或您配置的相应地址和端口)。
2.  导航到预测页面 (通常是 "Start Prediction" 或类似链接)。
3.  您可以直接粘贴一个或多个氨基酸序列，或者上传一个 FASTA 格式的文件。
4.  提交任务后，系统将异步处理请求。您可以查看任务状态，并在完成后获取溶解度、热稳定性和毒性的预测结果。
5.  结果页面通常提供详细的预测值和置信度，并允许将结果下载为 CSV 文件。

## 注意事项

*   **GPU/内存管理:** 蛋白质语言模型和深度学习模型的训练/推理会消耗大量计算资源。请确保您的环境有足够的 RAM (和 VRAM, 如果使用 GPU)。Web 应用中的模型管理器包含一定的自动释放机制。
*   **模型路径配置:** 无论是本地运行还是 Docker 部署，都需要正确配置模型文件的路径。推荐使用环境变量结合代码中的默认路径 (如 Hugging Face Hub ID) 来管理。
*   **大型文件:** 本项目使用 Git LFS 管理大型数据文件和部分模型文件。克隆仓库后请确保已安装并运行 `git lfs pull`。

## 贡献

欢迎对此项目做出贡献！请通过提交 Pull Request 或创建 Issue 的方式参与。

## 许可证

(请在此处添加项目的许可证信息，例如 MIT, Apache 2.0 等。如果未定，可以暂时留空或写 "待定"。)

---
*中文版说明请参见 [README_zh.md](./README_zh.md)* 