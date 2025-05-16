"""
蛋白质语言模型训练框架 - 模块化设计
支持 S-PLM、ESMC、ESM2 三种模型的单独或融合训练
可以灵活配置不同训练模式和模型组合
"""
# 修改后的train_7_1.py文件，增加编码层。
import os
import gc
import json
import random
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoTokenizer, AutoModel, AutoConfig
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_curve, auc,
    precision_score, recall_score, f1_score, confusion_matrix, 
    matthews_corrcoef as mcc_score
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# 设置多进程启动方式为 'spawn'
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 已经设置过

# 禁用不必要的警告
warnings.filterwarnings('ignore')

#===============================================================================
# 配置模块
#===============================================================================

class ModelConfig:
    """模型配置基类"""
    def __init__(self, model_name, model_path=None, enabled=True):
        self.model_name = model_name
        self.model_path = model_path
        self.output_dim = None  # 模型输出维度
        self.enabled = enabled  # 是否启用该模型
        self.device = None      # 模型所在设备
    
    def to_dict(self):
        """将配置转换为字典，用于保存"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "output_dim": self.output_dim,
            "enabled": self.enabled
        }

class ESM2Config(ModelConfig):
    """ESM2模型配置"""
    def __init__(self, model_path="/HOME/scz0brz/run/model/esm2_t33_650M_UR50D", enabled=True):
        super().__init__("esm2", model_path, enabled)
        self.output_dim = 1280
    
    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({"type": "ESM2Config"})
        return base_dict

class ESMCConfig(ModelConfig):
    """ESM-C模型配置"""
    def __init__(self, model_path="esmc_600m", enabled=True):
        super().__init__("esmc", model_path, enabled)
        self.output_dim = 1152
    
    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({"type": "ESMCConfig"})
        return base_dict

class SPLMConfig(ModelConfig):
    """S-PLM模型配置"""
    def __init__(self, config_path="./configs/representation_config.yaml", 
                 checkpoint_path="/HOME/scz0brz/run/AA_solubility/model/checkpoint_0520000.pth", 
                 enabled=True):
        super().__init__("splm", None, enabled)
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.output_dim = 1280
    
    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            "type": "SPLMConfig",
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path
        })
        return base_dict

class TrainingConfig:
    """训练配置"""
    def __init__(self):
        # 数据路径
        self.train_csv = "model_1_data.csv"
        self.test_csv = "toxin_test_filtered.csv"
        self.target_column = "label"  # 目标列名
        self.sequence_column = "sequence"  # 序列列名
        
        # 模型保存路径
        self.model_save_dir = "./protein_model_results_3_2"
        
        # 训练参数
        self.batch_size = 16
        self.epochs = 20
        self.lr = 5e-5
        self.weight_decay = 1e-6
        self.max_seq_len = 1024
        
        # 模型参数
        self.hidden_dim = 512
        self.dropout = 0.2
        
        # 训练模式
        self.train_mode = "fusion"  # 'fusion', 'single', 'ensemble'
        
        # 特征参数
        self.normalize_features = True
        self.feature_cache_size = 4000

        # 特征归一化参数
        self.normalize_features = True  # 是否启用特征归一化
        self.normalization_method = "global"  # 归一化方法: "none", "global", "sequence", "layer"
        # 预计算的统计值 (将在首次运行数据集时填充)
        self.esm2_mean = 0.0
        self.esm2_std = 1.0
        self.esmc_mean = 0.0
        self.esmc_std = 1.0
        self.splm_mean = 0.0
        self.splm_std = 1.0

        # 训练模式
        self.train_mode = "fusion"  # 'fusion', 'single', 'ensemble'
        self.fusion_type = "default"  # 'default', 'weighted', 'concat'

        # 训练设置
        self.use_amp = True  # 混合精度训练
        self.grad_clip = 1.0
        self.num_workers = 2
        self.num_folds = 5  # 5折交叉验证
        self.random_seed = 42
        self.warmup_ratio = 0.1
        self.patience = 10  # 早停
        self.class_weights = [1.0, 9.0]  # 处理类别不平衡
        self.negative_sampling_ratio = 10 # 阴性样本欠采样比例 (负:正)
        
        # GPU设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_gpu = torch.cuda.device_count() > 1
        self.use_separate_gpus = True  # 特征提取模型和训练模型使用不同GPU

        # 集成方法
        self.ensemble_method = "average"  # 集成方法: 'average', 'weighted', 'voting', 'max'

        
        if torch.cuda.device_count() >= 2:
            # 默认：特征提取GPU 0，训练GPU 1
            self.feature_extraction_device = torch.device("cuda:0")
            self.training_device = torch.device("cuda:1")
        else:
            # 只有一个GPU时共用
            self.feature_extraction_device = self.device
            self.training_device = self.device
    
    def set_seed(self):
        """设置随机种子以确保可重现性"""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def to_dict(self):
        """将配置转换为字典，用于保存"""
        return {k: v if not isinstance(v, torch.device) else str(v) 
                for k, v in self.__dict__.items()}

class ExperimentConfig:
    """实验配置，整合训练配置和模型配置"""
    def __init__(self, name="default_experiment"):
        self.name = name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_config = TrainingConfig()
        
        # 默认包含所有模型
        self.model_configs = {
            "esm2": ESM2Config(),
            "esmc": ESMCConfig(),
            "splm": SPLMConfig(enabled=False)  # 默认禁用S-PLM，因为需要额外依赖
        }
    
    def get_run_dir(self):
        """获取运行目录"""
        base_dir = self.training_config.model_save_dir
        return os.path.join(base_dir, f"{self.name}_{self.timestamp}")
    
    def save_config(self, filepath):
        """保存配置到JSON文件"""
        config_dict = {
            "name": self.name,
            "timestamp": self.timestamp,
            "training_config": self.training_config.to_dict(),
            "model_configs": {k: v.to_dict() for k, v in self.model_configs.items()}
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict):
        """从字典加载配置"""
        instance = cls(config_dict.get("name", "loaded_experiment"))
        instance.timestamp = config_dict.get("timestamp", instance.timestamp)
        
        # 加载训练配置
        for k, v in config_dict.get("training_config", {}).items():
            setattr(instance.training_config, k, v)
        
        # 加载模型配置
        instance.model_configs = {}
        for model_name, model_config in config_dict.get("model_configs", {}).items():
            if model_config["type"] == "ESM2Config":
                instance.model_configs[model_name] = ESM2Config(
                    model_path=model_config.get("model_path"),
                    enabled=model_config.get("enabled", True)
                )
            elif model_config["type"] == "ESMCConfig":
                instance.model_configs[model_name] = ESMCConfig(
                    model_path=model_config.get("model_path"),
                    enabled=model_config.get("enabled", True)
                )
            elif model_config["type"] == "SPLMConfig":
                instance.model_configs[model_name] = SPLMConfig(
                    config_path=model_config.get("config_path"),
                    checkpoint_path=model_config.get("checkpoint_path"),
                    enabled=model_config.get("enabled", True)
                )
        
        return instance
    
    @classmethod
    def load_config(cls, filepath):
        """从JSON文件加载配置"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)

#===============================================================================
# 日志模块
#===============================================================================

class Logger:
    """日志管理类"""
    def __init__(self, log_file=None, console=True):
        self.log_file = log_file
        self.console = console
        
        # 创建日志目录
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log(self, message, level="INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}"
        
        # 打印到控制台
        if self.console:
            print(formatted_message)
        
        # 写入日志文件
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted_message + "\n")
    
    def info(self, message):
        self.log(message, "INFO")
    
    def warning(self, message):
        self.log(message, "WARNING")
    
    def error(self, message):
        self.log(message, "ERROR")
    
    def debug(self, message):
        self.log(message, "DEBUG")

#===============================================================================
# 数据加载与预处理模块
#===============================================================================

class SequenceDataset:
    """蛋白质序列数据集加载与预处理"""
    
    @staticmethod
    def load_from_csv(file_path, sequence_col="sequence", target_col="label", logger=None):
        """从CSV文件加载数据"""
        try:
            df = pd.read_csv(file_path)
            if logger:
                logger.info(f"成功加载数据集: {file_path}, 共 {len(df)} 条记录")
            return df
        except Exception as e:
            if logger:
                logger.error(f"加载数据集失败: {e}")
            return None
    
    @staticmethod
    def get_data_stats(df, target_col="label"):
        """获取数据集统计信息"""
        stats = {}
        stats["total_count"] = len(df)
        
        # 目标分布
        if target_col in df.columns:
            stats["target_distribution"] = df[target_col].value_counts().to_dict()
            
        # 序列长度分布
        if "sequence" in df.columns:
            seq_lengths = df["sequence"].str.len()
            stats["seq_length"] = {
                "min": seq_lengths.min(),
                "max": seq_lengths.max(),
                "mean": seq_lengths.mean(),
                "median": seq_lengths.median()
            }
        
        return stats

#===============================================================================
# 特征提取模块
#===============================================================================

class FeatureExtractor:
    """特征提取基类"""
    def __init__(self, config, device=None, logger=None):
        self.config = config
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.logger = logger
    
    def load_model(self):
        """加载模型"""
        raise NotImplementedError("子类必须实现此方法")
    
    def extract_features(self, sequence, max_len):
        """提取特征"""
        raise NotImplementedError("子类必须实现此方法")
    
    def cleanup(self):
        """清理资源"""
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            self.model = None
            if self.logger:
                self.logger.info(f"{self.config.model_name} 模型资源已释放")

class ESM2Extractor(FeatureExtractor):
    """ESM2特征提取器"""
    def __init__(self, config, device=None, logger=None):
        super().__init__(config, device, logger)
        self.tokenizer = None
    
    def load_model(self):
        """加载ESM2模型"""
        if self.model is not None:
            return self.model
        
        try:
            if self.logger:
                self.logger.info(f"加载ESM2模型: {self.config.model_path}")
                
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            
            # 配置模型
            model_config = AutoConfig.from_pretrained(self.config.model_path, output_hidden_states=True)
            
            # 关闭dropout以确保推理结果确定性
            model_config.hidden_dropout = 0.
            model_config.hidden_dropout_prob = 0.
            model_config.attention_dropout = 0.
            model_config.attention_probs_dropout_prob = 0.
            
            # 加载模型
            self.model = AutoModel.from_pretrained(
                self.config.model_path, 
                config=model_config
            ).to(self.device).eval()
            
            if self.logger:
                self.logger.info(f"ESM2模型加载成功，设备: {self.device}")
            
            return self.model
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"ESM2模型加载失败: {e}")
            return None
    
    def extract_features(self, sequence, max_len):
        """从ESM2模型提取特征"""
        if self.model is None:
            self.load_model()
            
        with torch.no_grad():
            # 在氨基酸之间添加空格
            spaced_seq = " ".join(list(sequence))
            
            # 编码序列
            inputs = self.tokenizer.encode_plus(
                spaced_seq, 
                return_tensors=None, 
                add_special_tokens=True,
                padding=True,
                truncation=True
            )
            
            # 转换为tensor并移至设备
            for k, v in inputs.items():
                inputs[k] = torch.tensor(v, dtype=torch.long).unsqueeze(0).to(self.device)
            
            # 前向传播
            outputs = self.model(input_ids=inputs['input_ids'], 
                                attention_mask=inputs['attention_mask'])
            
            # 提取最后一层隐藏状态
            last_hidden_states = outputs[0]
            
            # 提取有效token的嵌入 (跳过首尾特殊标记)
            encoded_seq = last_hidden_states[0, inputs['attention_mask'][0].bool()][1:-1].cpu()
            
            # 处理序列长度
            current_len = encoded_seq.shape[0]
            if current_len < max_len:
                # 填充
                pad_len = max_len - current_len
                padded_residue = torch.zeros((max_len, encoded_seq.size(1)))
                padded_residue[:current_len] = encoded_seq
                padded_mask = torch.zeros(max_len, dtype=torch.bool)
                padded_mask[:current_len] = True
            elif current_len > max_len:
                # 截断
                padded_residue = encoded_seq[:max_len]
                padded_mask = torch.ones(max_len, dtype=torch.bool)
            else:
                padded_residue = encoded_seq
                padded_mask = torch.ones(max_len, dtype=torch.bool)
                
            # 计算整体表示（平均池化）
            global_representation = encoded_seq.mean(dim=0)
                
            return padded_residue, padded_mask, global_representation

class ESMCExtractor(FeatureExtractor):
    """ESM-C特征提取器"""
    _shared_model = None
    _load_lock = mp.Lock()
    
    def load_model(self):
        """加载ESM-C模型，加入共享机制和错误重试"""
        # 如果已有共享模型，直接使用
        if ESMCExtractor._shared_model is not None:
            self.model = ESMCExtractor._shared_model
            return self.model
            
        if self.model is not None:
            return self.model
            
        # 使用互斥锁防止多进程同时加载
        with ESMCExtractor._load_lock:
            # 双重检查，可能在获取锁的过程中已被其他进程加载
            if ESMCExtractor._shared_model is not None:
                self.model = ESMCExtractor._shared_model
                return self.model
                
            # 先清理GPU缓存
            torch.cuda.empty_cache()
            
            try:
                if self.logger:
                    self.logger.info(f"加载ESM-C模型: {self.config.model_path}")
                    
                from esm.models.esmc import ESMC
                
                # 尝试在GPU上加载
                self.model = ESMC.from_pretrained(self.config.model_path).to(self.device).eval()
                
                # 设置为共享模型
                ESMCExtractor._shared_model = self.model
                
                if self.logger:
                    self.logger.info(f"ESM-C模型加载成功，设备: {self.device}")
                    
                return self.model
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and self.logger:
                    self.logger.warning(f"GPU内存不足，尝试在CPU上加载ESM-C模型: {e}")
                    
                    try:
                        # 在CPU上加载
                        self.model = ESMC.from_pretrained(self.config.model_path).cpu().eval()
                        
                        # 设置为共享模型
                        ESMCExtractor._shared_model = self.model
                        
                        if self.logger:
                            self.logger.info("ESM-C模型在CPU上加载成功")
                            
                        return self.model
                    except Exception as cpu_e:
                        if self.logger:
                            self.logger.error(f"ESM-C模型加载失败: {cpu_e}")
                        return None
                        
                elif self.logger:
                    self.logger.error(f"ESM-C模型加载失败: {e}")
                return None
    
    def extract_features(self, sequence, max_len):
        """从ESM-C模型提取特征"""
        # 确保模型已加载，但不重复加载过程的日志输出
        if self.model is None:
            self.load_model()
            
        # 如果模型加载失败，返回备用特征
        if self.model is None:
            if self.logger:
                self.logger.warning(f"ESM-C模型未加载，返回备用特征")
            dummy_features = torch.zeros((max_len, self.config.output_dim))
            dummy_mask = torch.zeros(max_len, dtype=torch.bool)
            dummy_global = torch.zeros(self.config.output_dim)
            return dummy_features, dummy_mask, dummy_global
            
        with torch.no_grad():
            try:
                # 确保当前操作在适当的设备上进行
                device = next(self.model.parameters()).device
                
                from esm.sdk.api import ESMProtein, LogitsConfig
                
                # 准备蛋白质数据
                protein = ESMProtein(sequence=sequence)
                protein_tensor = self.model.encode(protein).to(device)
                
                # 获取特征
                logits_output = self.model.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True)
                )
                
                # 提取并处理嵌入特征，去除首尾标记
                embeddings = logits_output.embeddings[0][1:-1].cpu()
                
                # 处理长度
                current_len = embeddings.shape[0]
                if current_len < max_len:
                    # 填充
                    padded_residue = torch.zeros((max_len, embeddings.size(1)))
                    padded_residue[:current_len] = embeddings
                    padded_mask = torch.zeros(max_len, dtype=torch.bool)
                    padded_mask[:current_len] = True
                elif current_len > max_len:
                    # 截断
                    padded_residue = embeddings[:max_len]
                    padded_mask = torch.ones(max_len, dtype=torch.bool)
                else:
                    padded_residue = embeddings
                    padded_mask = torch.ones(max_len, dtype=torch.bool)
                
                # 计算整体表示
                global_representation = embeddings.mean(dim=0)
                
                return padded_residue, padded_mask, global_representation
                    
            except Exception as e:
                if self.logger:
                    self.logger.error(f"ESM-C特征提取失败: {e}")
                    
                # 返回备用特征
                dummy_features = torch.zeros((max_len, self.config.output_dim))
                dummy_mask = torch.zeros(max_len, dtype=torch.bool)
                dummy_mask[:min(50, max_len)] = True  # 假设序列长度为50
                dummy_global = torch.zeros(self.config.output_dim)
                return dummy_features, dummy_mask, dummy_global

# 修改 SPLMExtractor 类以匹配您现有的特征提取逻辑

class SPLMExtractor(FeatureExtractor):
    """S-PLM特征提取器"""
    def load_model(self):
        """加载S-PLM模型"""
        if self.model is not None:
            return self.model
            
        try:
            # 加载S-PLM模型
            if self.logger:
                self.logger.info(f"加载S-PLM模型: {self.config.checkpoint_path}")
                
            try:
                # 根据 splm_extract_4.py 中的实现调整
                import yaml
                from utils import load_configs, load_checkpoints_only
                from model import SequenceRepresentation
                
                # 加载配置文件
                try:
                    # 如果是字符串路径，直接加载文件
                    if isinstance(self.config.config_path, str):
                        with open(self.config.config_path) as file:
                            dict_config = yaml.full_load(file)
                        configs = load_configs(dict_config)
                    else:
                        # 如果已经是字典，直接使用
                        configs = load_configs(self.config.config_path)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"加载S-PLM配置文件失败: {e}")
                    return None
                
                # 创建模型
                model = SequenceRepresentation(logging=None, configs=configs)
                model.to(self.device)
                
                # 加载检查点
                load_checkpoints_only(self.config.checkpoint_path, model)
                model.eval()  # 设置为评估模式
                
                self.model = model
                
                if self.logger:
                    self.logger.info(f"S-PLM模型加载成功，设备: {self.device}")
                    
                return self.model
            except ImportError as e:
                if self.logger:
                    self.logger.error(f"加载S-PLM模型所需的模块未找到，请确保相关依赖已安装: {e}")
                return None
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"S-PLM模型加载失败: {e}")
            return None
    
    def extract_features(self, sequence, max_len):
        """从S-PLM模型提取特征，基于 splm_extract_4.py 的实现"""
        # 尝试加载模型
        if self.model is None:
            self.load_model()
            
        # 如果模型仍然为 None，返回备用特征
        if self.model is None:
            if self.logger:
                self.logger.warning(f"使用S-PLM备用特征，模型未成功加载")
                
            # 返回备用特征
            dummy_features = torch.zeros((max_len, self.config.output_dim))
            dummy_mask = torch.zeros(max_len, dtype=torch.bool)
            dummy_mask[:min(50, max_len)] = True  # 假设序列长度为50
            dummy_global = torch.zeros(self.config.output_dim)
            return dummy_features, dummy_mask, dummy_global
        
        try:
            with torch.no_grad():
                # 准备序列
                esm2_seq = [(range(len(sequence)), str(sequence))]
                
                # 使用模型的转换器
                batch_labels, batch_strs, batch_tokens = self.model.batch_converter(esm2_seq)
                
                # 获取输入 token
                token = batch_tokens.to(self.device)
                
                # 填充/截断到合适长度
                if token.size(1) < max_len:
                    padding = torch.ones((1, max_len - token.size(1)), dtype=token.dtype, device=self.device) * self.model.alphabet.padding_idx
                    token = torch.cat([token, padding], dim=1)
                elif token.size(1) > max_len:
                    token = token[:, :max_len]  # 截断过长序列
                
                # 获取蛋白质表示、残基表示和掩码
                protein_representation, residue_representation, mask = self.model(token)
                
                # 处理输出
                global_repr = protein_representation.squeeze(0).cpu()  # 全局表示
                residue_repr = residue_representation.squeeze(0).cpu()  # 残基表示
                attention_mask = mask.squeeze(0).cpu().bool()  # 注意力掩码
                
                # 确保长度匹配
                if residue_repr.size(0) != max_len:
                    if residue_repr.size(0) < max_len:
                        # 填充
                        padded_residue = torch.zeros((max_len, residue_repr.size(1)), device=residue_repr.device)
                        padded_residue[:residue_repr.size(0)] = residue_repr
                        padded_mask = torch.zeros(max_len, dtype=torch.bool, device=attention_mask.device)
                        padded_mask[:attention_mask.size(0)] = attention_mask
                        
                        residue_repr = padded_residue
                        attention_mask = padded_mask
                    else:
                        # 截断
                        residue_repr = residue_repr[:max_len]
                        attention_mask = attention_mask[:max_len]
                
                return residue_repr, attention_mask, global_repr
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"S-PLM特征提取失败: {e}")
                
            # 返回备用特征
            dummy_features = torch.zeros((max_len, self.config.output_dim))
            dummy_mask = torch.zeros(max_len, dtype=torch.bool)
            dummy_mask[:min(50, max_len)] = True  # 假设序列长度为50
            dummy_global = torch.zeros(self.config.output_dim)
            return dummy_features, dummy_mask, dummy_global
    
    def tokenize_sequence(self, sequence):
        """将序列转换为token_ids和attention_mask"""
        # 这是一个示例，实际情况需要根据S-PLM的tokenizer调整
        # 假设我们有一个简单的氨基酸到id的映射
        aa_to_id = {aa: i+1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        aa_to_id["<pad>"] = 0
        
        # 转换序列
        token_ids = [aa_to_id.get(aa, 0) for aa in sequence]
        attention_mask = [1] * len(token_ids)
        
        # 转换为tensor
        token_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long)
        
        return token_ids, attention_mask

class FeatureManager:
    """特征管理器，用于管理多种特征提取器"""
    def __init__(self, config, logger=None):
        self.config = config
        self.extractors = {}
        self.feature_device = config.training_config.feature_extraction_device
        self.logger = logger
        self.loaded_models = {}  # 添加模型缓存
        self.models_loaded = False  # 添加标记，记录模型是否已加载
        # 注册已启用的模型配置
        for name, model_config in self.config.model_configs.items():
            if model_config.enabled:
                self.register_extractor(name, model_config)
    
    def register_extractor(self, name, extractor_config):
        """注册特征提取器"""
        if name == "esm2":
            self.extractors[name] = ESM2Extractor(extractor_config, self.feature_device, self.logger)
        elif name == "esmc":
            self.extractors[name] = ESMCExtractor(extractor_config, self.feature_device, self.logger)
        elif name == "splm":
            self.extractors[name] = SPLMExtractor(extractor_config, self.feature_device, self.logger)
        else:
            if self.logger:
                self.logger.warning(f"不支持的特征提取器类型: {name}")
            raise ValueError(f"不支持的特征提取器类型: {name}")
    
    def extract_all_features(self, sequence, max_len):
        """提取所有已注册特征提取器的特征"""
        # 确保所有模型已加载
        if not self.models_loaded:
            self.preload_models()
            
        features = {}
        for name, extractor in self.extractors.items():
            residue_repr, mask, global_repr = extractor.extract_features(sequence, max_len)
            features[name] = (residue_repr, mask, global_repr)
            
            # 在每个特征提取后释放GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return features
        
    def cleanup(self):
        """清理资源"""
        for extractor in self.extractors.values():
            extractor.cleanup()
    def preload_models(self):
        """预热加载所有模型，确保只加载一次"""
        if self.models_loaded:
            return
        
        if self.logger:
            self.logger.info("预热加载所有特征提取模型...")
            
        for name, extractor in self.extractors.items():
            if self.logger:
                self.logger.info(f"预热加载 {name} 模型...")
            extractor.load_model()
            
        self.models_loaded = True
        if self.logger:
            self.logger.info("所有特征提取模型加载完成，将在整个训练过程中保持加载状态")
    
    def extract_all_features(self, sequence, max_len):
        """提取所有已注册特征提取器的特征"""
        # 确保所有模型已加载
        if not self.models_loaded:
            self.preload_models()
            
        features = {}
        for name, extractor in self.extractors.items():
            residue_repr, mask, global_repr = extractor.extract_features(sequence, max_len)
            features[name] = (residue_repr, mask, global_repr)
        return features
#===============================================================================
# 数据集与数据加载模块
#===============================================================================

class ProteinFeatureDataset(Dataset):
    """蛋白质特征数据集"""
    def __init__(self, df, feature_manager, config, target_col="label", 
                 sequence_col="sequence", cache_size=100, logger=None):
        """
        初始化数据集
        
        参数:
            df: DataFrame包含序列和标签
            feature_manager: 特征管理器
            config: 配置对象
            target_col: 目标列名
            sequence_col: 序列列名
            cache_size: 特征缓存大小
            logger: 日志记录器
        """
        self.df = df
        self.feature_manager = feature_manager
        self.config = config
        self.target_col = target_col
        self.sequence_col = sequence_col
        self.logger = logger
        
        # 特征缓存
        self.feature_cache = {}
        self.cache_size = cache_size
        self.cache_keys = []
        # 增强特征缓存，使用序列作为键而不是索引，这样相同序列就不会重复提取特征
        self.sequence_to_features = {}

    def __len__(self):
        return len(self.df)
    
    def _update_cache(self, idx, features):
        """更新LRU缓存"""
        if len(self.cache_keys) >= self.cache_size:
            # 移除最早使用的项
            oldest = self.cache_keys.pop(0)
            if oldest in self.feature_cache:
                del self.feature_cache[oldest]
        
        # 添加新项
        self.cache_keys.append(idx)
        self.feature_cache[idx] = features
    
    def _normalize_features(self, features, feature_type):
    
        """根据配置的方法归一化特征"""
        # 如果禁用归一化，直接返回原始特征
        if not self.config.training_config.normalize_features:
            return features
            
        # 基于方法选择归一化策略
        if self.config.training_config.normalization_method == "global":
            # 使用全局均值和标准差
            if feature_type == "esm2":
                return (features - self.config.training_config.esm2_mean) / self.config.training_config.esm2_std
            elif feature_type == "esmc":
                return (features - self.config.training_config.esmc_mean) / self.config.training_config.esmc_std
            elif feature_type == "splm":
                return (features - self.config.training_config.splm_mean) / self.config.training_config.splm_std
                
        elif self.config.training_config.normalization_method == "sequence":
            # 对每个序列单独归一化
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True) + 1e-6
            return (features - mean) / std
            
        # 默认情况：不归一化
        return features

# 改进 ProteinFeatureDataset 的 __getitem__ 方法

    def __getitem__(self, idx):
        """获取单个样本的特征"""
        # 提取序列和标签
        seq = self.df.iloc[idx][self.sequence_col]
        
        if self.target_col in self.df.columns:
            label = float(self.df.iloc[idx][self.target_col])
        else:
            # 如果没有标签列（如测试集），则使用-1作为占位符
            label = -1.0
        
        # 使用序列缓存，避免重复计算
        if seq in self.sequence_to_features:
            features = self.sequence_to_features[seq]
        # 然后检查索引缓存
        elif idx in self.feature_cache:
            # 使用缓存的特征
            features, _ = self.feature_cache[idx]
            
            # 更新缓存顺序
            self.cache_keys.remove(idx)
            self.cache_keys.append(idx)
        else:
            # 计算序列最大长度
            max_len = min(len(seq), self.config.training_config.max_seq_len)
            
            try:
                # 提取特征
                features = self.feature_manager.extract_all_features(seq, max_len)
                
                # 更新缓存，并管理缓存大小
                if len(self.cache_keys) >= self.cache_size:
                    # 一次清除多个旧缓存项，而不是只清除一个
                    to_remove = min(5, len(self.cache_keys) // 10)  # 清除约10%的缓存
                    for _ in range(to_remove):
                        if self.cache_keys:
                            oldest = self.cache_keys.pop(0)
                            if oldest in self.feature_cache:
                                del self.feature_cache[oldest]
                
                # 添加新项
                self._update_cache(idx, (features, label))
                self.sequence_to_features[seq] = features
                
                # 主动清理过大的序列缓存
                if len(self.sequence_to_features) > self.cache_size * 2:
                    # 只保留最近使用的序列
                    recent_keys = set(self.cache_keys[-self.cache_size:])
                    recent_seqs = {self.df.iloc[k][self.sequence_col] for k in recent_keys if k < len(self.df)}
                    self.sequence_to_features = {s: f for s, f in self.sequence_to_features.items() if s in recent_seqs}
                    gc.collect()  # 手动触发垃圾回收
                    
            except Exception as e:
                if self.logger:
                    self.logger.error(f"提取序列 {idx} 特征时出错: {e}")
                # 返回备用特征
                features = {}
                for name in self.feature_manager.extractors.keys():
                    dim = self.feature_manager.extractors[name].config.output_dim
                    dummy_residue = torch.zeros((max_len, dim))
                    dummy_mask = torch.zeros(max_len, dtype=torch.bool)
                    dummy_global = torch.zeros(dim)
                    features[name] = (dummy_residue, dummy_mask, dummy_global)
        
        return features, label


def collate_protein_features(batch):
    """
    批处理函数，处理不同长度的序列

    batch结构:
    [(features_dict_1, label_1), (features_dict_2, label_2), ...]
    
    features_dict结构:
    {
        'esm2': (residue_repr, mask, global_repr),
        'esmc': (residue_repr, mask, global_repr),
        ...
    }
    """
    features_dict, labels = zip(*batch)
    result = {'labels': torch.tensor(labels, dtype=torch.float)}
    
    # 获取特征名列表
    feature_names = list(features_dict[0].keys())
    
    # 对于混合精度训练，统一使用 float32，模型内部会自动转换
    dtype = torch.float32
    
    for name in feature_names:
        # 提取当前特征类型的所有批次数据
        try:
            residue_reprs = [item[name][0] for item in features_dict]
            masks = [item[name][1] for item in features_dict]
            global_reprs = [item[name][2] for item in features_dict]
            
            # 获取当前批次最大序列长度
            max_len = max(feat.size(0) for feat in residue_reprs)
            feat_dim = residue_reprs[0].size(1)
            
            # 填充到相同长度
            padded_reprs = []
            padded_masks = []
            
            for i in range(len(residue_reprs)):
                curr_len = residue_reprs[i].size(0)
                
                # 创建填充张量，使用统一数据类型
                residue_pad = torch.zeros(max_len, feat_dim, dtype=dtype)
                mask_pad = torch.zeros(max_len, dtype=torch.bool)
                
                # 复制数据
                residue_pad[:curr_len] = residue_reprs[i].to(dtype)
                mask_pad[:curr_len] = masks[i]
                
                padded_reprs.append(residue_pad)
                padded_masks.append(mask_pad)
            
            # 添加到结果，所有特征使用统一数据类型
            result[f"{name}_residue"] = torch.stack(padded_reprs)
            result[f"{name}_mask"] = torch.stack(padded_masks)
            result[f"{name}_global"] = torch.stack([g.to(dtype) for g in global_reprs])
        except Exception as e:
            # 处理可能的KeyError或其他错误
            print(f"处理特征 {name} 时出错: {e}")
            continue
    return result

#===============================================================================
# 模型定义模块
#===============================================================================

class SingleModelClassifier(nn.Module):
    """单一蛋白质语言模型分类器"""
    def __init__(self, input_dim, hidden_dim=512, dropout=0.2, model_name="esm2"):
        super().__init__()
        self.model_name = model_name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 特征归一化层
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # 增加深度的特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 预测头
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
# 修复 SingleModelClassifier 类的 forward 方法

    def forward(self, batch):
        """
        前向传播
        
        参数:
            batch: 可以是字典或直接是特征元组
            如果是字典，应该包含键 "{model_name}_residue", "{model_name}_mask", "{model_name}_global"
        """
        # 识别输入类型并处理
        if isinstance(batch, dict):
            # 检查批次中是否包含有效的模型特征
            available_keys = list(batch.keys())
            
            # 首先，尝试使用 self.model_name 查找特征
            model_name = getattr(self, 'model_name', None)
            
            if model_name:
                residue_key = f"{model_name}_residue"
                mask_key = f"{model_name}_mask" 
                global_key = f"{model_name}_global"
                
                if residue_key in batch or global_key in batch:
                    # 使用预先定义的模型名称
                    residue_repr = batch.get(residue_key)
                    mask = batch.get(mask_key)
                    global_repr = batch.get(global_key)
                else:
                    # 如果找不到指定模型的特征，尝试从可用键中查找任何模型特征
                    model_found = False
                    for key in available_keys:
                        if key.endswith('_residue'):
                            model_name = key.replace('_residue', '')
                            residue_key = key
                            mask_key = f"{model_name}_mask"
                            global_key = f"{model_name}_global"
                            
                            residue_repr = batch.get(residue_key)
                            mask = batch.get(mask_key)
                            global_repr = batch.get(global_key)
                            model_found = True
                            
                            # 更新模型名称以供将来使用
                            self.model_name = model_name
                            break
                    
                    if not model_found:
                        # 如果所有尝试都失败，抛出错误
                        raise ValueError(f"无法从batch中找到有效的模型特征。可用键: {available_keys}")
            else:
                # 如果模型名称未设置，尝试查找任何可用的模型特征
                model_found = False
                for key in available_keys:
                    if key.endswith('_residue'):
                        model_name = key.replace('_residue', '')
                        residue_key = key
                        mask_key = f"{model_name}_mask"
                        global_key = f"{model_name}_global"
                        
                        residue_repr = batch.get(residue_key)
                        mask = batch.get(mask_key)
                        global_repr = batch.get(global_key)
                        model_found = True
                        
                        # 保存模型名称以供将来使用
                        self.model_name = model_name
                        break
                
                if not model_found:
                    # 如果所有尝试都失败，抛出错误
                    raise ValueError(f"无法从batch中找到有效的模型特征。可用键: {available_keys}")
        else:
            # 假设输入是 (residue_repr, mask, global_repr) 元组
            if isinstance(batch, tuple) and len(batch) >= 2:
                residue_repr = batch[0]
                mask = batch[1] if len(batch) > 1 else None
                global_repr = batch[2] if len(batch) > 2 else None
            else:
                # 直接将输入视为残基表示
                residue_repr = batch
                mask = None
                global_repr = None
        
        # 如果没有全局表示，则从残基表示计算
        if global_repr is None and residue_repr is not None:
            if mask is not None:
                # 使用mask进行池化
                mask = mask.unsqueeze(-1).float()
                valid_tokens = mask.sum(dim=1).clamp(min=1)
                x = (residue_repr * mask).sum(dim=1) / valid_tokens
            else:
                # 简单平均池化
                x = residue_repr.mean(dim=1)
        else:
            x = global_repr
        
        # 确保x的数据类型与layer_norm权重一致
        if x.dtype != self.layer_norm.weight.dtype:
            x = x.to(self.layer_norm.weight.dtype)
        
        # 特征归一化
        x = self.layer_norm(x)
        
        # 使用编码器处理特征
        x = self.encoder(x)
        
        # 使用预测头进行分类
        x = self.head(x)
        
        return x.squeeze(-1)  # 返回logits

    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型权重和配置信息
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'model_name': self.model_name
        }, path)
    
    @classmethod
    def load_model(cls, path, device="cuda"):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        
        # 创建模型实例
        model = cls(
            input_dim=checkpoint.get('input_dim', 1280),
            hidden_dim=checkpoint.get('hidden_dim', 512),
            model_name=checkpoint.get('model_name', 'esm2')
        ).to(device)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
class WeightedFusionClassifier(nn.Module):
    """加权融合模型分类器，适合相似维度的特征"""
    def __init__(self, model_configs, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.model_configs = model_configs
        self.hidden_dim = hidden_dim
        self.model_names = list(model_configs.keys())
        
        # 为每个模型创建特征归一化层
        self.layer_norms = nn.ModuleDict({
            name: nn.LayerNorm(config.output_dim)
            for name, config in model_configs.items()
        })
        
        # 为每个模型创建特征投影层
        self.projections = nn.ModuleDict({
            name: nn.Linear(config.output_dim, hidden_dim)
            for name, config in model_configs.items()
        })
        
        # 统一的特征编码层 - 新增
        self.feature_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 可学习的特征权重
        self.feature_weights = nn.Parameter(
            torch.ones(len(model_configs)) / len(model_configs)
        )
        
        # 增强的预测头
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
        )
    
    def forward(self, batch):
        # 处理每个模型的输入并投影到相同维度
        projected_features = []
        
        for model_name in self.model_names:
            # 从输入获取该模型的特征
            global_key = f"{model_name}_global"
            residue_key = f"{model_name}_residue"
            mask_key = f"{model_name}_mask"
            
            if global_key in batch:
                # 使用预计算的全局表示
                global_repr = batch[global_key]
            else:
                # 从残基表示计算全局表示
                residue_repr = batch[residue_key]
                mask = batch.get(mask_key)
                
                if mask is not None:
                    mask = mask.unsqueeze(-1).float()
                    valid_tokens = mask.sum(dim=1).clamp(min=1)
                    global_repr = (residue_repr * mask).sum(dim=1) / valid_tokens
                else:
                    global_repr = residue_repr.mean(dim=1)
            
            # 特征归一化
            norm_repr = self.layer_norms[model_name](global_repr)
            
            # 投影
            proj_repr = self.projections[model_name](norm_repr)
            
            # 应用统一的编码层 - 新增
            encoded_repr = self.feature_encoder(proj_repr)
            
            projected_features.append(encoded_repr)
        
        # 加权融合
        stacked_features = torch.stack(projected_features, dim=1)  # [B, num_models, D]
        weights = F.softmax(self.feature_weights, dim=0)  # [num_models]
        weighted_sum = (stacked_features * weights.view(1, -1, 1)).sum(dim=1)  # [B, D]
        
        # 融合特征归一化和预测
        x = self.fusion_norm(weighted_sum)
        x = self.head(x)
        
        return x.squeeze(-1)  # 返回logits
    
    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型权重和配置信息
        model_configs_dict = {
            name: {
                "output_dim": config.output_dim
            }
            for name, config in self.model_configs.items()
        }
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'feature_weights': self.feature_weights.data.tolist(),
            'hidden_dim': self.hidden_dim,
            'model_configs': model_configs_dict,
            'model_names': self.model_names
        }, path)
    
    @classmethod
    def load_model(cls, path, device="cuda"):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        
        # 重建模型配置
        model_configs = {}
        for name, config_dict in checkpoint['model_configs'].items():
            model_config = ModelConfig(name)
            model_config.output_dim = config_dict["output_dim"]
            model_configs[name] = model_config
        
        # 创建模型实例
        model = cls(
            model_configs=model_configs,
            hidden_dim=checkpoint.get('hidden_dim', 512)
        ).to(device)
        
        # 加载权重
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f"Successfully loaded state dict for {cls.__name__} from {path} with strict=True.")
        except RuntimeError as e:
            # 如果严格加载失败，尝试非严格加载并发出警告
            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                print(f"⚠️ Warning: State dict mismatch for {cls.__name__} at {path}. "
                      f"Attempting to load with strict=False. This might indicate an older checkpoint format. "
                      f"Layers like 'feature_encoder' or parts of 'head' might be re-initialized if they differ.")
                # Log the specific missing/unexpected keys for debugging
                print(f"   Details: {e}") 
                try:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print(f"Successfully loaded state dict for {cls.__name__} from {path} with strict=False.")
                except Exception as load_err:
                    print(f"❌ Error: Failed to load state dict even with strict=False for {path}. Error: {load_err}")
                    raise load_err # Re-raise the error if non-strict loading also fails
            else:
                # 如果是其他 RuntimeError，则重新抛出
                print(f"❌ Error: Unexpected RuntimeError during state dict loading for {path}: {e}")
                raise e
        
        return model
    
class FusionModelClassifier(nn.Module):
    """融合多个蛋白质语言模型的分类器"""
    def __init__(self, model_configs, hidden_dim=512, dropout=0.1):
        """
        初始化融合模型
        
        参数:
            model_configs: 字典，键为模型名，值为模型配置
        """
        super().__init__()
        self.model_configs = model_configs
        self.hidden_dim = hidden_dim
        self.model_names = list(model_configs.keys())
        
        # 为每个模型创建特征归一化层
        self.layer_norms = nn.ModuleDict({
            name: nn.LayerNorm(config.output_dim)
            for name, config in model_configs.items()
        })
        
        # 为每个模型创建特征投影层
        self.projections = nn.ModuleDict({
            name: nn.Linear(config.output_dim, hidden_dim)
            for name, config in model_configs.items()
        })
        
        # 可学习的特征权重
        self.feature_weights = nn.Parameter(
            torch.ones(len(model_configs)) / len(model_configs)
        )
        
        # 预测头
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
    
# 修复 FusionModelClassifier 的 forward 方法

# 修复 FusionModelClassifier 的 forward 方法中的属性访问

    def forward(self, batch):
        """
        前向传播
        
        参数:
            batch: 包含所有特征的字典
            {
                "esm2_residue": tensor,
                "esm2_mask": tensor,
                "esm2_global": tensor,
                "esmc_residue": tensor,
                ...
                "labels": tensor
            }
        """
        # 处理每个模型的输入并投影到相同维度
        projected_features = []
        
        for model_name in self.model_configs.keys():
            # 从输入获取该模型的特征
            residue_key = f"{model_name}_residue"
            mask_key = f"{model_name}_mask"
            global_key = f"{model_name}_global"
            
            if global_key in batch:
                # 使用预计算的全局表示
                global_repr = batch[global_key]
            else:
                # 从残基表示计算全局表示
                residue_repr = batch[residue_key]
                mask = batch.get(mask_key)
                
                if mask is not None:
                    # 使用mask进行池化
                    mask = mask.unsqueeze(-1).float()
                    valid_tokens = mask.sum(dim=1).clamp(min=1)
                    global_repr = (residue_repr * mask).sum(dim=1) / valid_tokens
                else:
                    # 简单平均池化
                    global_repr = residue_repr.mean(dim=1)
            
            # 确保数据类型一致
            layer_norm = self.layer_norms[model_name]
            if global_repr.dtype != layer_norm.weight.dtype:
                global_repr = global_repr.to(layer_norm.weight.dtype)
                
            # 特征归一化
            norm_repr = layer_norm(global_repr)
            
            # 投影
            proj_repr = self.projections[model_name](norm_repr)
            projected_features.append(proj_repr)
        
        # 计算加权融合
        stacked_features = torch.stack(projected_features, dim=1)  # [B, num_models, D]
        weights = F.softmax(self.feature_weights, dim=0)  # [num_models]
        
        # 应用权重
        weighted_sum = (stacked_features * weights.view(1, -1, 1)).sum(dim=1)  # [B, D]
        x = self.fusion_norm(weighted_sum)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)  # 添加额外的dropout层增强正则化
        x = self.fc2(x)
        
        return x.squeeze(-1)  # 返回logits
        
    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型权重和配置信息
        model_configs_dict = {
            name: {
                "output_dim": config.output_dim
            }
            for name, config in self.model_configs.items()
        }
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'feature_weights': self.feature_weights.data.tolist(),
            'hidden_dim': self.hidden_dim,
            'model_configs': model_configs_dict,
            'model_names': self.model_names
        }, path)
    
    @classmethod
    def load_model(cls, path, device="cuda"):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        
        # 重建模型配置
        model_configs = {}
        for name, config_dict in checkpoint['model_configs'].items():
            model_config = ModelConfig(name)
            model_config.output_dim = config_dict["output_dim"]
            model_configs[name] = model_config
        
        # 创建模型实例
        model = cls(
            model_configs=model_configs,
            hidden_dim=checkpoint.get('hidden_dim', 512)
        ).to(device)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model

#===============================================================================
# 训练与评估模块
#===============================================================================

class ModelTrainer:
    """模型训练与评估类"""
    def __init__(self, experiment_config, logger=None):
        """
        初始化训练器
        
        参数:
            experiment_config: 实验配置
            logger: 日志记录器
        """
        self.config = experiment_config
        self.training_config = experiment_config.training_config
        self.logger = logger
        
        # 设置随机种子
        self.training_config.set_seed()
        
        # 创建运行目录
        self.run_dir = experiment_config.get_run_dir()
        os.makedirs(self.run_dir, exist_ok=True)
        
        # 保存配置
        self.config.save_config(os.path.join(self.run_dir, "config.json"))
        
        # 初始化结果存储
        self.results = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": [],
            "test_metrics": [],
            "fold_models": [],
            "best_model_path": None,
            "best_val_mcc": -1.0
        }
        
        # 初始化特征管理器
        self.feature_manager = None
    
    def log(self, message):
        """记录日志"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def _init_feature_manager(self):
        """初始化特征管理器并预热模型"""
        if self.feature_manager is None:
            self.log("初始化特征管理器...")
            self.feature_manager = FeatureManager(self.config, self.logger)
            
            # 预热加载所有模型
            self.log("预热加载所有特征提取模型...")
            self.feature_manager.preload_models()
    
    def _create_model(self):
        """创建模型"""
        train_mode = self.training_config.train_mode
        
        if train_mode == "fusion":
            # 创建融合模型
            return self._create_fusion_model()
        elif train_mode == "single":
            # 只使用单个模型
            return self._create_single_model()
        else:
            raise ValueError(f"不支持的训练模式: {train_mode}")
    
    def _create_fusion_model(self):
        """创建融合模型"""
        # 获取启用的模型配置
        enabled_models = {
            name: config for name, config in self.config.model_configs.items() 
            if config.enabled
        }
        
        if not enabled_models:
            raise ValueError("没有启用的模型配置，无法创建融合模型")
        
        # 根据fusion_type创建不同的融合模型
        fusion_type = self.training_config.fusion_type
        
        if fusion_type == "weighted":
            model = WeightedFusionClassifier(
                model_configs=enabled_models,
                hidden_dim=self.training_config.hidden_dim,
                dropout=self.training_config.dropout
            ).to(self.training_config.training_device)
            self.log(f"已创建加权融合模型，融合 {list(enabled_models.keys())} 模型特征")
        else:
            # 默认使用FusionModelClassifier
            model = FusionModelClassifier(
                model_configs=enabled_models,
                hidden_dim=self.training_config.hidden_dim,
                dropout=self.training_config.dropout
            ).to(self.training_config.training_device)
            self.log(f"已创建默认融合模型，融合 {list(enabled_models.keys())} 模型特征")
        
        return model
        
    def _create_single_model(self):
        """创建单模型"""
        # 查找第一个启用的模型
        for name, config in self.config.model_configs.items():
            if config.enabled:
                # 使用这个模型
                model = SingleModelClassifier(
                    input_dim=config.output_dim,
                    hidden_dim=self.training_config.hidden_dim,
                    dropout=self.training_config.dropout,
                    model_name=name  # 确保传入模型名称
                ).to(self.training_config.training_device)
                
                self.log(f"已创建单模型分类器，使用 {name} 模型特征")
                return model
        
        raise ValueError("没有启用的模型配置，无法创建单模型")
        
    def train_kfold(self):
        """K折交叉验证训练"""
        self._init_feature_manager()
        
        # 加载数据 (修改)
        try:
            train_pos_df = pd.read_csv(self.training_config.train_pos_csv)
            train_neg_df = pd.read_csv(self.training_config.train_neg_csv)
            test_pos_df = pd.read_csv(self.training_config.test_pos_csv)
            test_neg_df = pd.read_csv(self.training_config.test_neg_csv)
            
            self.log(f"成功加载训练集阳性数据: {self.training_config.train_pos_csv} ({len(train_pos_df)}条)")
            self.log(f"成功加载训练集阴性数据: {self.training_config.train_neg_csv} ({len(train_neg_df)}条)")
            self.log(f"成功加载测试集阳性数据: {self.training_config.test_pos_csv} ({len(test_pos_df)}条)")
            self.log(f"成功加载测试集阴性数据: {self.training_config.test_neg_csv} ({len(test_neg_df)}条)")

            # 添加标签
            train_pos_df['label'] = 1
            train_neg_df['label'] = 0
            test_pos_df['label'] = 1
            test_neg_df['label'] = 0
            
            # 合并数据
            train_df = pd.concat([train_pos_df, train_neg_df], ignore_index=True)
            test_df = pd.concat([test_pos_df, test_neg_df], ignore_index=True)
            
            # 打乱数据
            train_df = train_df.sample(frac=1, random_state=self.training_config.random_seed).reset_index(drop=True)
            test_df = test_df.sample(frac=1, random_state=self.training_config.random_seed).reset_index(drop=True)
            
            # 重命名列 (确保统一)
            train_df = train_df.rename(columns={'Sequence': self.training_config.sequence_column, 
                                                'label': self.training_config.target_column})
            test_df = test_df.rename(columns={'Sequence': self.training_config.sequence_column, 
                                               'label': self.training_config.target_column})
            
            # 选择需要的列
            train_df = train_df[[self.training_config.sequence_column, self.training_config.target_column]]
            test_df = test_df[[self.training_config.sequence_column, self.training_config.target_column]]

            self.log(f"合并后训练集大小: {len(train_df)}")
            self.log(f"合并后测试集大小: {len(test_df)}")

        except FileNotFoundError as e:
            self.log(f"错误: 文件未找到 - {e}. 请检查文件路径.")
            return None, None
        except Exception as e:
            self.log(f"加载数据时发生错误: {e}")
            return None, None

        # 移除旧的加载方式
        # train_df = SequenceDataset.load_from_csv(...) 
        # test_df = SequenceDataset.load_from_csv(...)

        if train_df is None or test_df is None:
            self.log("数据加载失败，终止训练")
            return None, None
        
        # 获取数据统计信息
        train_stats = SequenceDataset.get_data_stats(train_df, self.training_config.target_column)
        test_stats = SequenceDataset.get_data_stats(test_df, self.training_config.target_column)
        
        self.log(f"训练集统计: {json.dumps(train_stats, indent=2)}")
        self.log(f"测试集统计: {json.dumps(test_stats, indent=2)}")
        
        # 创建K折交叉验证
        kfold = StratifiedKFold(
            n_splits=self.training_config.num_folds,
            shuffle=True,
            random_state=self.training_config.random_seed
        )

        # 加载数据后，在使用kfold.split之前添加
        # 检查标签中的NaN值
        nan_mask = pd.isna(train_df[self.training_config.target_column])
        if nan_mask.any():
            self.log(f"警告：发现{nan_mask.sum()}个样本包含NaN标签，这些样本将被删除")
            train_df = train_df[~nan_mask].reset_index(drop=True)
        # 准备分层K折
        X = train_df[self.training_config.sequence_column].values
        y = train_df[self.training_config.target_column].values
        
        # 存储每折结果
        fold_results = []
        test_predictions = []
        fold_model_paths = []
        # 训练每一折
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
            self.log(f"\n{'='*50}\n开始训练第 {fold}/{self.training_config.num_folds} 折\n{'='*50}")
            
            # 创建数据子集
            # fold_train_df = train_df.iloc[train_idx].reset_index(drop=True) # 原来的训练集
            # fold_val_df = train_df.iloc[val_idx].reset_index(drop=True) # 原来的验证集

            # 1. 验证集 (Validation Set): 不平衡，直接使用当前折数据
            fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)
            val_pos_count = fold_val_df[self.training_config.target_column].sum()
            val_neg_count = len(fold_val_df) - val_pos_count
            self.log(f"  Fold {fold} - 验证集 (不平衡): {len(fold_val_df)} 条 (阳性: {val_pos_count}, 阴性: {val_neg_count})")

            # 2. 训练集池 (Training Pool): 合并剩余折
            training_pool_df = train_df.iloc[train_idx].reset_index(drop=True)
            pool_pos_count = training_pool_df[self.training_config.target_column].sum()
            pool_neg_count = len(training_pool_df) - pool_pos_count
            self.log(f"  Fold {fold} - 训练集池: {len(training_pool_df)} 条 (阳性: {pool_pos_count}, 阴性: {pool_neg_count})")

            # 3. 平衡训练集 (Balanced Training Set): 欠采样
            positive_samples = training_pool_df[training_pool_df[self.training_config.target_column] == 1]
            negative_samples = training_pool_df[training_pool_df[self.training_config.target_column] == 0]
            
            num_positives = len(positive_samples)
            num_negatives_to_sample = int(num_positives * self.training_config.negative_sampling_ratio)
            num_negatives_to_sample = min(num_negatives_to_sample, len(negative_samples)) # 确保不超过可用数量

            sampled_negative_samples = negative_samples.sample(n=num_negatives_to_sample, random_state=self.training_config.random_seed)
            
            # 合并形成平衡训练集
            fold_train_df = pd.concat([positive_samples, sampled_negative_samples]).sample(frac=1, random_state=self.training_config.random_seed).reset_index(drop=True)
            train_pos_count = fold_train_df[self.training_config.target_column].sum()
            train_neg_count = len(fold_train_df) - train_pos_count
            self.log(f"  Fold {fold} - 训练集 (平衡, 欠采样比例 1:{self.training_config.negative_sampling_ratio}): {len(fold_train_df)} 条 (阳性: {train_pos_count}, 阴性: {train_neg_count})")
            
            # 创建数据集 (使用平衡训练集和不平衡验证集)
            train_dataset = ProteinFeatureDataset(
                fold_train_df,  # 使用平衡的训练集
                self.feature_manager,
                self.config,
                target_col=self.training_config.target_column,
                sequence_col=self.training_config.sequence_column,
                cache_size=self.training_config.feature_cache_size,
                logger=self.logger
            )
            
            val_dataset = ProteinFeatureDataset(
                fold_val_df,  # 使用不平衡的验证集
                self.feature_manager,
                self.config,
                target_col=self.training_config.target_column,
                sequence_col=self.training_config.sequence_column,
                cache_size=self.training_config.feature_cache_size // 2,
                logger=self.logger
            )
            
            test_dataset = ProteinFeatureDataset(
                test_df,
                self.feature_manager,
                self.config,
                target_col=self.training_config.target_column,
                sequence_col=self.training_config.sequence_column,
                cache_size=self.training_config.feature_cache_size // 2,
                logger=self.logger
            )
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=True,
                num_workers=self.training_config.num_workers,
                collate_fn=collate_protein_features,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=self.training_config.num_workers,
                collate_fn=collate_protein_features,
                pin_memory=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=self.training_config.num_workers,
                collate_fn=collate_protein_features,
                pin_memory=True
            )
            
            # 创建模型
            model = self._create_model()
            
            # 训练当前折
            fold_result, fold_model = self._train_fold(model, train_loader, val_loader, test_loader, fold)
            fold_results.append(fold_result)
            
            # 保存模型路径
            self.results["fold_models"].append(fold_result["model_path"])
            
            # 收集测试集预测
            test_predictions.append(fold_result["test_predictions"])
            fold_model_paths.append(fold_result["model_path"])
            # 清理内存
            del model, train_dataset, val_dataset, test_dataset
            del train_loader, val_loader, test_loader
            gc.collect()
            torch.cuda.empty_cache()
        
        # 创建测试数据集
        test_dataset = ProteinFeatureDataset(
            test_df,
            self.feature_manager,
            self.config,
            target_col=self.training_config.target_column,
            sequence_col=self.training_config.sequence_column,
            cache_size=self.training_config.feature_cache_size // 2,
            logger=self.logger
        )
        
        # 创建测试数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=self.training_config.num_workers,
            collate_fn=collate_protein_features,
            pin_memory=True
        )
        
        # 使用集成方法进行预测
        self.log(f"\n{'='*50}\n使用{self.training_config.ensemble_method}集成方法进行测试集预测\n{'='*50}")
        ensemble_predictions = self._ensemble_predictions(
            test_loader, 
            fold_model_paths, 
            ensemble_method=self.training_config.ensemble_method
        )
        
        # 计算二分类标签
        ensemble_labels = (ensemble_predictions >= 0.5).astype(int)
        
        # 计算集成测试指标
        if self.training_config.target_column in test_df.columns:
            test_true = test_df[self.training_config.target_column].values
            ensemble_metrics = self._calculate_metrics(test_true, ensemble_labels, ensemble_predictions)
            self.results["test_metrics"] = ensemble_metrics
            
            self.log(f"\n{'='*50}")
            self.log(f"集成模型测试集性能 (方法: {self.training_config.ensemble_method}):")
            for k, v in ensemble_metrics.items():
                if isinstance(v, (int, float)):
                    self.log(f"{k}: {v:.4f}")
            self.log(f"{'='*50}\n")
        else:
            ensemble_metrics = {"predictions": ensemble_predictions, "labels": ensemble_labels}
    
        
        # 绘制结果图表
        if self.training_config.target_column in test_df.columns:
            self._plot_final_results(fold_results, ensemble_metrics, test_df)
        
        # 返回结果
        return fold_results, ensemble_metrics
    
    def _train_fold(self, model, train_loader, val_loader, test_loader, fold):
        """训练单折模型"""
        # 初始化结果字典
        fold_result = {
            "fold": fold,
            "train_loss": [],
            "val_loss": [],
            "val_metrics": [],
            "best_epoch": 0,
            "best_val_loss": float('inf'),
            "best_val_mcc": -1.0,
            "model_path": None,
            "test_predictions": None
        }
        
        # 设置优化器
        optimizer = AdamW(
            model.parameters(),
            lr=self.training_config.lr,
            weight_decay=self.training_config.weight_decay
        )
        
        
        # 设置学习率调度器
        total_steps = len(train_loader) * self.training_config.epochs
        warmup_steps = int(total_steps * self.training_config.warmup_ratio)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.training_config.lr,
            total_steps=total_steps,
            pct_start=self.training_config.warmup_ratio,
            div_factor=25,
            final_div_factor=1000
        )
        
        # 设置损失函数
        if self.training_config.class_weights:
            pos_weight = torch.tensor([self.training_config.class_weights[1] / self.training_config.class_weights[0]]).to(self.training_config.training_device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # 混合精度训练
        scaler = GradScaler() if self.training_config.use_amp else None
        
        # 早停计数器
        patience_counter = 0
        best_val_mcc = -1.0
        
        # 开始训练
        for epoch in range(1, self.training_config.epochs + 1):
            # 训练模式
            model.train()
            train_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.training_config.epochs}")
            
            for batch in progress_bar:
                # 将数据移至设备
                batch = {k: v.to(self.training_config.training_device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                labels = batch["labels"]
                
                optimizer.zero_grad()
                
                # 使用混合精度
                if self.training_config.use_amp:
                    with autocast():
                        outputs = model(batch)
                        loss = criterion(outputs, labels)
                    
                    # 缩放损失并反向传播
                    scaler.scale(loss).backward()
                    
                    # 梯度裁剪
                    if self.training_config.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.training_config.grad_clip)
                    
                    # 更新权重
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 标准训练流程
                    outputs = model(batch)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
                    # 梯度裁剪
                    if self.training_config.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.training_config.grad_clip)
                    
                    optimizer.step()
                
                # 更新学习率
                scheduler.step()
                
                # 更新进度条
                train_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{train_loss/(progress_bar.n+1):.4f}"})
            
            # 计算平均训练损失
            avg_train_loss = train_loss / len(train_loader)
            fold_result["train_loss"].append(avg_train_loss)
            
            # 验证
            val_loss, val_metrics = self._validate(model, val_loader, criterion)
            fold_result["val_loss"].append(val_loss)
            fold_result["val_metrics"].append(val_metrics)
            
            # 记录日志
            log_message = (f"Epoch {epoch}/{self.training_config.epochs} - "
                           f"Train Loss: {avg_train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}")
            
            # 添加所有验证指标到日志
            metrics_log = []
            for metric_name, metric_value in val_metrics.items():
                if isinstance(metric_value, (int, float)):
                    metrics_log.append(f"Val {metric_name.upper()}: {metric_value:.4f}")
                else:
                    # 对于非数值指标（如混淆矩阵元素），直接打印
                    metrics_log.append(f"Val {metric_name.upper()}: {metric_value}")
            
            log_message += ", " + ", ".join(metrics_log)
            self.log(log_message)
            # 每个epoch结束后主动清理内存
            torch.cuda.empty_cache()
            gc.collect()

            # --- Modification Start ---
            # Save model checkpoint for the current epoch
            epoch_model_path = os.path.join(self.run_dir, f"fold_{fold}_epoch_{epoch}.pt")
            model.save_model(epoch_model_path)
            # self.log(f"Saved epoch {epoch} checkpoint to: {epoch_model_path}") # Optional: uncomment to log epoch saves
            # --- Modification End ---

            # 检查是否为最佳模型
            if val_metrics['mcc'] > best_val_mcc:
                best_val_mcc = val_metrics['mcc']
                patience_counter = 0
                
                # 保存最佳模型
                model_path = os.path.join(self.run_dir, f"fold_{fold}_best_model.pt")
                model.save_model(model_path)
                
                # 更新结果
                fold_result["best_epoch"] = epoch
                fold_result["best_val_loss"] = val_loss
                fold_result["best_val_mcc"] = best_val_mcc
                fold_result["model_path"] = model_path
                
                # 更新全局最佳模型
                if best_val_mcc > self.results["best_val_mcc"]:
                    self.results["best_val_mcc"] = best_val_mcc
                    self.results["best_model_path"] = model_path
                
                self.log(f"✅ 新的最佳模型! MCC: {best_val_mcc:.4f}")
            else:
                patience_counter += 1
                
            # 早停检查
            if patience_counter >= self.training_config.patience:
                self.log(f"早停触发! {self.training_config.patience} 轮没有改善.")
                break
        
        # 加载最佳模型进行测试
        if fold_result["model_path"]:
            if isinstance(model, WeightedFusionClassifier):
                best_model = WeightedFusionClassifier.load_model(
                    fold_result["model_path"],
                    device=self.training_config.training_device
                )
            elif isinstance(model, FusionModelClassifier):
                best_model = FusionModelClassifier.load_model(
                    fold_result["model_path"],
                    device=self.training_config.training_device
                )
            else:
                best_model = SingleModelClassifier.load_model(
                    fold_result["model_path"],
                    device=self.training_config.training_device
                )
                
            # 在测试集上评估最佳模型
            test_preds, _ = self._evaluate_test(best_model, test_loader)
            fold_result["test_predictions"] = test_preds
            
            # 返回结果和模型
            return fold_result, best_model
        
        # 如果没有保存最佳模型，也返回结果
        return fold_result, None
# 继续实现 ModelTrainer 类的剩余方法
    def _validate(self, model, val_loader, criterion):
        """模型验证 - 优化内存使用"""
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        
        # 分批次验证，避免内存堆积
        batch_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 将数据移至设备
                batch = {k: v.to(self.training_config.training_device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                labels = batch["labels"]
                
                # 使用混合精度减少内存使用
                if self.training_config.use_amp:
                    with autocast():
                        outputs = model(batch)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(batch)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item()
                
                # 立即转为CPU numpy并释放GPU内存
                probs = torch.sigmoid(outputs).cpu().numpy()
                labels_cpu = labels.cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                
                all_labels.extend(labels_cpu)
                all_preds.extend(preds)
                all_probs.extend(probs)
                
                # 显式释放不再需要的张量，减少碎片
                del outputs, loss, labels, batch
                
                # 每处理10个批次清理一次缓存
                batch_count += 1
                if batch_count % 10 == 0:
                    torch.cuda.empty_cache()
        
        # 最终清理
        torch.cuda.empty_cache()
        
        # 计算平均损失
        avg_val_loss = val_loss / len(val_loader)
        
        # 转换为numpy数组再计算指标，减少内存使用
        metrics = self._calculate_metrics(
            np.array(all_labels), 
            np.array(all_preds), 
            np.array(all_probs)
        )
        
        return avg_val_loss, metrics
    
    def _evaluate_test(self, model, test_loader):
        """评估测试集 - 内存优化版本"""
        model.eval()
        all_labels = []
        all_probs = []
        batch_count = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # 将数据移至设备
                batch = {k: v.to(self.training_config.training_device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 使用混合精度减少内存使用
                if self.training_config.use_amp:
                    with autocast():
                        outputs = model(batch)
                else:
                    outputs = model(batch)
                
                # 立即转为CPU numpy并释放GPU张量
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.extend(probs)
                
                # 如果有标签，也收集它们
                if "labels" in batch:
                    labels_cpu = batch["labels"].cpu().numpy()
                    all_labels.extend(labels_cpu)
                
                # 显式释放不再需要的张量
                del outputs, batch
                
                # 每处理10个批次清理一次缓存
                batch_count += 1
                if batch_count % 10 == 0:
                    torch.cuda.empty_cache()
        
        # 最终清理
        torch.cuda.empty_cache()
        
        # 转换为numpy数组
        all_probs_array = np.array(all_probs)
        all_preds = (all_probs_array >= 0.5).astype(int)
        
        # 如果有标签，计算指标
        if all_labels:
            metrics = self._calculate_metrics(
                np.array(all_labels), 
                all_preds, 
                all_probs_array
            )
            return all_probs_array, metrics
        else:
            return all_probs_array, None
    
    def _calculate_metrics(self, true_labels, pred_labels, probabilities):
        """计算评估指标"""
        metrics = {}
        
        # 基本指标
        metrics["accuracy"] = accuracy_score(true_labels, pred_labels)
        metrics["precision"] = precision_score(true_labels, pred_labels, zero_division=0)
        metrics["recall"] = recall_score(true_labels, pred_labels, zero_division=0)
        metrics["f1"] = f1_score(true_labels, pred_labels, zero_division=0)
        metrics["mcc"] = mcc_score(true_labels, pred_labels)
        
        # AUC指标
        if len(np.unique(true_labels)) > 1:
            metrics["auc"] = roc_auc_score(true_labels, probabilities)
            precision, recall, _ = precision_recall_curve(true_labels, probabilities)
            metrics["pr_auc"] = auc(recall, precision)
        else:
            metrics["auc"] = 0.0
            metrics["pr_auc"] = 0.0
        
        # 混淆矩阵值
        tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels, labels=[0, 1]).ravel()
        metrics["tn"] = int(tn)
        metrics["fp"] = int(fp)
        metrics["fn"] = int(fn)
        metrics["tp"] = int(tp)
        
        return metrics
    
    def _calculate_average_metrics(self, metrics_list):
        """计算所有折的平均指标"""
        avg_metrics = {}
        metrics_keys = metrics_list[0].keys() if metrics_list else []
        
        for key in metrics_keys:
            # 只平均数字指标，不处理列表或其他类型
            if all(isinstance(m[key], (int, float)) for m in metrics_list):
                avg_metrics[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
        
        return avg_metrics

    def _ensemble_predictions(self, test_loader, fold_models, ensemble_method="average"):
        """
        使用多种集成方法对测试集进行预测
        
        参数:
            test_loader: 测试数据加载器
            fold_models: 各折训练的模型路径列表
            ensemble_method: 集成方法，可选值:
                - "average": 平均值集成
                - "weighted": 加权平均值集成（基于验证集MCC）
                - "voting": 投票集成（多数投票）
                - "max": 最大值集成
        
        返回:
            predictions: 预测结果
        """
        all_fold_predictions = []
        weights = []
        
        self.log(f"使用{ensemble_method}集成方法进行预测...")
        
        # 收集每个模型的预测
        for i, model_path in enumerate(fold_models):
            self.log(f"加载第{i+1}折模型: {model_path}")
            
            # 根据模型类型加载模型
            checkpoint = torch.load(model_path, map_location=self.training_config.training_device)
            
            if 'model_names' in checkpoint and isinstance(checkpoint['model_names'], list) and len(checkpoint['model_names']) > 1:
                if "feature_weights" in checkpoint:
                    model = WeightedFusionClassifier.load_model(
                        model_path,
                        device=self.training_config.training_device
                    )
                else:
                    model = FusionModelClassifier.load_model(
                        model_path,
                        device=self.training_config.training_device
                    )
            else:
                model = SingleModelClassifier.load_model(
                    model_path,
                    device=self.training_config.training_device
                )
            
            # 获取模型在验证集上的性能权重
            val_mcc = float(checkpoint.get('best_val_mcc', 0.0))
            if val_mcc <= 0:  # 确保权重为正
                val_mcc = 0.1
            weights.append(val_mcc)
            
            # 获取测试集预测
            fold_preds, _ = self._evaluate_test(model, test_loader)
            all_fold_predictions.append(fold_preds)
            
            # 释放模型
            del model
            torch.cuda.empty_cache()
        
        # 确保所有预测转换为numpy数组
        all_fold_predictions = [np.array(preds) for preds in all_fold_predictions]
        
        # 开始集成
        if ensemble_method == "average":
            # 平均值集成
            ensemble_preds = np.mean(all_fold_predictions, axis=0)
            self.log(f"使用平均值集成：{len(all_fold_predictions)}个模型")
            
        elif ensemble_method == "weighted":
            # 加权平均值集成（基于验证集性能）
            total_weight = sum(weights)
            normalized_weights = [w/total_weight for w in weights]
            
            self.log(f"使用加权平均集成，权重分别为: {normalized_weights}")
            
            # 加权求和
            ensemble_preds = np.zeros_like(all_fold_predictions[0])
            for i, preds in enumerate(all_fold_predictions):
                ensemble_preds += normalized_weights[i] * preds
                
        elif ensemble_method == "voting":
            # 投票集成（二分类问题）
            binary_preds = [(p >= 0.5).astype(int) for p in all_fold_predictions]
            vote_sum = np.sum(binary_preds, axis=0)
            # 多数票决定
            ensemble_preds = (vote_sum >= len(fold_models)/2).astype(float)
            
            self.log(f"使用投票集成：{len(all_fold_predictions)}个模型投票")
            
        elif ensemble_method == "max":
            # 最大值集成
            ensemble_preds = np.max(all_fold_predictions, axis=0)
            self.log(f"使用最大值集成：取{len(all_fold_predictions)}个模型中的最大预测值")
        
        else:
            self.log(f"不支持的集成方法: {ensemble_method}，使用平均值集成")
            ensemble_preds = np.mean(all_fold_predictions, axis=0)
        
        # 保存集成信息
        ensemble_info = {
            "method": ensemble_method,
            "fold_models": fold_models,
            "weights": weights if ensemble_method == "weighted" else None
        }
        self.results["ensemble_info"] = ensemble_info
        
        return ensemble_preds

    def _plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix", save_path=None):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Non-toxic', 'Toxic'],
            yticklabels=['Non-toxic', 'Toxic']
        )
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_final_results(self, fold_results, ensemble_metrics, test_df):
        """绘制最终结果图表"""
        # 创建图表目录
        plots_dir = os.path.join(self.run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. 绘制训练和验证损失曲线 (第一折)
        plt.figure(figsize=(10, 6))
        plt.plot(fold_results[0]["train_loss"], label='Training Loss', marker='o')
        plt.plot(fold_results[0]["val_loss"], label='Validation Loss', marker='x')
        plt.title('Training and Validation Loss (Fold 1)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(plots_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 绘制所有折的MCC值比较
        plt.figure(figsize=(10, 6))
        fold_mccs = [fold["val_metrics"][-1]["mcc"] for fold in fold_results]
        plt.bar(range(1, len(fold_mccs) + 1), fold_mccs, color='skyblue')
        plt.axhline(y=ensemble_metrics["mcc"], color='r', linestyle='--', label=f'Ensemble MCC: {ensemble_metrics["mcc"]:.4f}')
        plt.title('MCC Comparison Across Folds')
        plt.xlabel('Fold')
        plt.ylabel('MCC')
        plt.xticks(range(1, len(fold_mccs) + 1))
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(plots_dir, 'fold_mcc_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 绘制集成模型的混淆矩阵
        if "label" in test_df.columns:
            test_true = test_df["label"].values
            test_pred = (ensemble_metrics["predictions"] >= 0.5).astype(int)
            self._plot_confusion_matrix(
                test_true, 
                test_pred,
                title=f"Ensemble Model Confusion Matrix (MCC={ensemble_metrics['mcc']:.4f})",
                save_path=os.path.join(plots_dir, 'ensemble_confusion_matrix.png')
            )
        
        # 4. 保存集成模型的预测结果
        if isinstance(ensemble_metrics["predictions"], np.ndarray):
            pred_df = test_df.copy()
            pred_df["prediction_prob"] = ensemble_metrics["predictions"]
            pred_df["prediction"] = (ensemble_metrics["predictions"] >= 0.5).astype(int)
            pred_df.to_csv(os.path.join(self.run_dir, "ensemble_predictions.csv"), index=False)
        
        # 5. 保存训练结果摘要
        summary = {
            "fold_results": [
                {
                    "fold": fold["fold"],
                    "best_epoch": fold["best_epoch"],
                    "best_val_mcc": fold["best_val_mcc"],
                    "best_val_metrics": fold["val_metrics"][fold["best_epoch"] - 1]
                }
                for fold in fold_results
            ],
            "ensemble_metrics": ensemble_metrics
        }
        
        with open(os.path.join(self.run_dir, "results_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

#===============================================================================
# 命令行参数解析
#===============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="蛋白质语言模型训练框架")
    
    # 实验配置
    parser.add_argument("--name", type=str, default="protein_experiment",
                        help="实验名称")
    
    # 数据路径 (修改)
    parser.add_argument("--train_pos_csv", type=str, default="csm_toxin_0.7.csv",
                        help="训练集阳性样本CSV文件路径")
    parser.add_argument("--train_neg_csv", type=str, default="csm_notoxin_0.7.csv",
                        help="训练集阴性样本CSV文件路径")
    parser.add_argument("--test_pos_csv", type=str, default="filtered_toxin_0.7.csv",
                        help="测试集阳性样本CSV文件路径")
    parser.add_argument("--test_neg_csv", type=str, default="filtered_notoxin_0.7.csv",
                        help="测试集阴性样本CSV文件路径")

    # 模型保存路径 - 新增参数
    parser.add_argument("--save_dir", type=str, default="./protein_model_results_3_2",
                        help="模型保存目录")
    
    # 训练模式
    parser.add_argument("--mode", type=str, choices=["fusion", "single"], default="fusion",
                        help="训练模式：融合或单一模型")
    parser.add_argument("--fusion_type", type=str, choices=["default", "weighted"], default="default",
                        help="融合模型类型：默认或加权")
    # 训练配置
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批量大小")
    parser.add_argument("--epochs", type=int, default=20,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="学习率")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="隐藏层维度")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout比率")
    parser.add_argument("--max_seq_len", type=int, default=1024,
                        help="最大序列长度")
    parser.add_argument("--num_folds", type=int, default=5,
                        help="交叉验证折数")
    parser.add_argument("--feature_cache_size", type=int, default=10000,
                        help="特征缓存大小")
    parser.add_argument("--weight_decay", type=float, default=5e-6,
                        help="权重衰减")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载器工作线程数")
    parser.add_argument("--negative_sampling_ratio", type=int, default=10,
                        help="训练集负样本欠采样比例 (负:正)")
    # 集成方法
    parser.add_argument("--ensemble_method", type=str, 
                      choices=["average", "weighted", "voting", "max"], 
                      default="average",
                      help="集成模型方法: 平均值、加权平均、投票或最大值")

    
    # 模型选择
    parser.add_argument("--use_esm2", action="store_true", help="使用ESM2模型")
    parser.add_argument("--use_esmc", action="store_true", help="使用ESM-C模型")
    parser.add_argument("--use_splm", action="store_true", help="使用S-PLM模型")
    
    # 模型路径
    parser.add_argument("--esm2_path", type=str,
                       default="/HOME/scz0brz/run/model/esm2_t33_650M_UR50D",
                       help="ESM2模型路径")
    parser.add_argument("--esmc_path", type=str,
                       default="esmc_600m",
                       help="ESM-C模型路径")
    parser.add_argument("--splm_config", type=str,
                       default="./configs/representation_config.yaml",
                       help="S-PLM配置文件路径")
    parser.add_argument("--splm_checkpoint", type=str,
                       default="/HOME/scz0brz/run/AA_solubility/model/checkpoint_0520000.pth",
                       help="S-PLM检查点文件路径")
    
    # GPU设置
    parser.add_argument("--feature_device", type=int, default=0,
                       help="特征提取GPU ID")
    parser.add_argument("--training_device", type=int, default=1,
                       help="模型训练GPU ID")
    parser.add_argument("--no_separate_gpu", action="store_true",
                       help="不使用独立GPU进行特征提取和训练")

    return parser.parse_args()

#===============================================================================
# 主程序入口
#===============================================================================

def main():
    """主函数"""
    # 设置环境变量以减少内存碎片
    torch.cuda.memory._set_allocator_settings('expandable_segments:False')  # 添加此行
    # 解析命令行参数
    args = parse_args()
    
    # 创建实验配置
    config = ExperimentConfig(name=args.name)
    
    # 更新训练配置 (修改数据路径)
    config.training_config.train_pos_csv = args.train_pos_csv
    config.training_config.train_neg_csv = args.train_neg_csv
    config.training_config.test_pos_csv = args.test_pos_csv
    config.training_config.test_neg_csv = args.test_neg_csv
    # config.training_config.train_csv = args.train_csv # 移除旧参数
    # config.training_config.test_csv = args.test_csv # 移除旧参数
    config.training_config.model_save_dir = args.save_dir
    config.training_config.batch_size = args.batch_size
    config.training_config.epochs = args.epochs
    config.training_config.lr = args.lr
    config.training_config.hidden_dim = args.hidden_dim
    config.training_config.dropout = args.dropout
    config.training_config.max_seq_len = args.max_seq_len
    config.training_config.num_folds = args.num_folds
    config.training_config.train_mode = args.mode
    config.training_config.num_workers = args.num_workers
    config.training_config.negative_sampling_ratio = args.negative_sampling_ratio # 新增
    config.training_config.train_mode = args.mode
    config.training_config.fusion_type = args.fusion_type
    config.training_config.ensemble_method = args.ensemble_method
    config.training_config.feature_cache_size = args.feature_cache_size
    config.training_config.weight_decay = args.weight_decay
    # 配置GPU
    if torch.cuda.is_available():
        if not args.no_separate_gpu and torch.cuda.device_count() >= 2:
            config.training_config.feature_extraction_device = torch.device(f"cuda:{args.feature_device}")
            config.training_config.training_device = torch.device(f"cuda:{args.training_device}")
            config.training_config.use_separate_gpus = True
            print(f"使用GPU {args.feature_device} 进行特征提取，GPU {args.training_device} 进行训练")
        else:
            # 只有一个GPU或不使用独立GPU
            device = torch.device("cuda:0")
            config.training_config.feature_extraction_device = device
            config.training_config.training_device = device
            config.training_config.use_separate_gpus = False
            print(f"使用同一GPU进行特征提取和训练")
    else:
        print("未检测到GPU，使用CPU运行")
        config.training_config.feature_extraction_device = torch.device("cpu")
        config.training_config.training_device = torch.device("cpu")
        config.training_config.use_separate_gpus = False
    
    # 配置模型
    config.model_configs["esm2"].enabled = args.use_esm2
    config.model_configs["esm2"].model_path = args.esm2_path

    config.model_configs["esmc"].enabled = args.use_esmc
    config.model_configs["esmc"].model_path = args.esmc_path
    
    config.model_configs["splm"].enabled = args.use_splm
    config.model_configs["splm"].config_path = args.splm_config
    config.model_configs["splm"].checkpoint_path = args.splm_checkpoint
    
    # 如果没有指定任何模型，默认使用ESM2
    if not (args.use_esm2 or args.use_esmc or args.use_splm):
        print("未指定任何模型，默认启用ESM2")
        config.model_configs["esm2"].enabled = True

    # 创建日志记录器
    log_file = os.path.join(config.get_run_dir(), "training.log")
    logger = Logger(log_file=log_file, console=True)
    
    logger.info(f"实验名称: {config.name}")
    logger.info(f"训练模式: {config.training_config.train_mode}")
    logger.info(f"启用的模型: " + ", ".join([name for name, cfg in config.model_configs.items() if cfg.enabled]))
    if config.training_config.train_mode == "fusion":
        logger.info(f"融合模型类型: {config.training_config.fusion_type}")
    logger.info(f"启用的模型: " + ", ".join([name for name, cfg in config.model_configs.items() if cfg.enabled]))
    logger.info(f"训练集负样本欠采样比例 (负:正): 1:{config.training_config.negative_sampling_ratio}") # 新增日志
    
    # 检查S-PLM模型文件路径
    if args.use_splm:
        splm_config_exists = os.path.exists(args.splm_config)
        splm_checkpoint_exists = os.path.exists(args.splm_checkpoint)
        
        if not splm_config_exists:
            logger.warning(f"S-PLM配置文件不存在: {args.splm_config}")
        
        if not splm_checkpoint_exists:
            logger.warning(f"S-PLM检查点文件不存在: {args.splm_checkpoint}")
        
        # 即使文件不存在也继续，因为我们有备用特征生成
        config.model_configs["splm"].enabled = True
        config.model_configs["splm"].config_path = args.splm_config
        config.model_configs["splm"].checkpoint_path = args.splm_checkpoint
        
        logger.info(f"已启用S-PLM模型，路径: {args.splm_checkpoint}")
    else:
        config.model_configs["splm"].enabled = False
    # 创建并执行训练器
    trainer = ModelTrainer(config, logger)
    logger.info("开始预热加载特征提取模型，这将确保模型只加载一次...")
    fold_results, ensemble_metrics = trainer.train_kfold()
  
    # 清理特征提取模型
    if trainer.feature_manager:
        trainer.feature_manager.cleanup()
    
    logger.info("\n" + "="*50)
    logger.info("✅ 训练完成!")
    if ensemble_metrics and "mcc" in ensemble_metrics:
        logger.info(f"最终测试集MCC: {ensemble_metrics['mcc']:.4f}")
        logger.info(f"最终测试集AUC: {ensemble_metrics['auc']:.4f}")
    logger.info("="*50)
    
    # 显示结果路径
    logger.info(f"结果保存在: {config.get_run_dir()}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 启动蛋白质语言模型训练框架")
    print("="*50 + "\n")
    
    main()