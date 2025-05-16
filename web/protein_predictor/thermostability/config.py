"""
蛋白质热稳定性预测配置类
"""
import os
import json
import torch
import numpy as np
import logging
from pathlib import Path

class ModelConfig:
    """模型配置类"""
    def __init__(self, **kwargs):
        self.enabled = kwargs.get('enabled', True)
        self.output_dim = kwargs.get('output_dim', 1280)
        self.model_name = kwargs.get('model_name', None)

class PredictorConfig:
    """预测器配置类"""
    def __init__(self, model_dir=None):
        # 基本配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 8
        self.use_amp = True
        
        # 模型目录
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "models", "thermostability"
        )
        
        # 特征统计
        self.esm2_mean = 0.0
        self.esm2_std = 1.0
        self.esmc_mean = 0.0
        self.esmc_std = 1.0
        
        # 模型参数
        self.seed = 42
        self.hidden_dim = 256  # 默认值，将从配置文件中覆盖
        self.dropout = 0.3     # 默认值，将从配置文件中覆盖
        
        # 如果模型目录存在，尝试从配置文件读取更多参数
        if os.path.exists(self.model_dir):
            config_path = os.path.join(self.model_dir, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        config_data = json.load(f)
                        
                    # 从训练配置中读取参数
                    train_config = config_data.get("training_config", {})
                    self.hidden_dim = train_config.get("hidden_dim", 768)
                    self.dropout = train_config.get("dropout", 0.3)
                    self.batch_size = train_config.get("batch_size", 16)
                    self.seed = train_config.get("random_seed", 42)
                except Exception as e:
                    logging.warning(f"读取配置文件失败: {e}，使用默认配置")
        
    def set_seed(self):
        """设置随机种子以确保可重复性"""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)