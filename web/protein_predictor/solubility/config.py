"""
蛋白质溶解性预测配置类
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
        self.batch_size = 16  # 从results.json获取的默认批量大小
        self.use_amp = True
        
        # 模型目录
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "models", "solubility"
        )
        
        # 特征维度
        self.esm2_dim = 1280
        self.esmc_dim = 1152
        
        # 特征统计
        self.esm2_mean = 0.0
        self.esm2_std = 1.0
        self.esmc_mean = 0.0
        self.esmc_std = 1.0
        
        # 特征归一化
        self.normalize_features = True
        self.normalization_method = "sequence"  # 从results.json获取的方法
        
        # 模型参数
        self.seed = 42
        self.hidden_dim = 256  # 从results.json获取
        self.dropout = 0.5     # 从results.json获取
        
        # 如果模型目录存在，尝试从配置文件读取更多参数
        if os.path.exists(self.model_dir):
            results_path = os.path.join(self.model_dir, "results.json")
            if os.path.exists(results_path):
                try:
                    with open(results_path, "r") as f:
                        results_data = json.load(f)
                        
                    # 从结果文件读取超参数
                    hyperparams = results_data.get("hyperparameters", {})
                    self.hidden_dim = hyperparams.get("hidden_dim", 256)
                    self.dropout = hyperparams.get("dropout", 0.5)
                    self.batch_size = hyperparams.get("batch_size", 16)
                    self.normalization_method = hyperparams.get("normalization_method", "sequence")
                    
                    logging.info(f"从results.json读取模型参数: hidden_dim={self.hidden_dim}, "
                                f"dropout={self.dropout}, normalization={self.normalization_method}")
                except Exception as e:
                    logging.warning(f"读取结果文件失败: {e}，使用默认配置")
        
    def set_seed(self):
        """设置随机种子以确保可重复性"""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)