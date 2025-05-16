"""
蛋白质特性预测共享数据处理模块
"""
import os
import torch
import numpy as np
import logging
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    """特征数据集"""
    def __init__(self, esm2_dir, esmc_dir, config, sample_ids=None):
        self.esm2_dir = esm2_dir
        self.esmc_dir = esmc_dir
        self.config = config
        
        # 查找共有样本
        esm2_files = {f.split('_features')[0]: f for f in os.listdir(esm2_dir) if f.endswith("_features.npy")}
        esmc_files = {f.split('_features')[0]: f for f in os.listdir(esmc_dir) if f.endswith("_features.npy")}
        common_ids = sorted(list(set(esm2_files.keys()) & set(esmc_files.keys())))
        
        # 筛选指定的样本ID（如果提供）
        if sample_ids:
            common_ids = [id for id in common_ids if id in sample_ids]
            
        self.common_ids = common_ids
        self.esm2_files = esm2_files
        self.esmc_files = esmc_files
        
    def __len__(self):
        return len(self.common_ids)
        
    def __getitem__(self, idx):
        """获取单个样本的特征"""
        sample_id = self.common_ids[idx]
        
        # 加载特征
        esm2_data = np.load(os.path.join(self.esm2_dir, self.esm2_files[sample_id]), allow_pickle=True).item()
        esmc_data = np.load(os.path.join(self.esmc_dir, self.esmc_files[sample_id]), allow_pickle=True).item()
        
        # 获取特征和掩码
        esm2_features = esm2_data["residue_representation"]
        esmc_features = esmc_data["residue_representation"]
        mask = esm2_data["mask"]
        
        # 标准化特征
        esm2_features = (esm2_features - self.config.esm2_mean) / self.config.esm2_std
        esmc_features = (esmc_features - self.config.esmc_mean) / self.config.esmc_std
        
        return {
            "sample_id": sample_id,
            "esm2_features": esm2_features,
            "esmc_features": esmc_features,
            "mask": mask
        }

def feature_collate_fn(batch):
    """特征批处理函数"""
    # 获取最大序列长度
    max_len = max([item["esm2_features"].shape[0] for item in batch])
    
    # 初始化批处理张量
    batch_size = len(batch)
    esm2_dim = batch[0]["esm2_features"].shape[1]
    esmc_dim = batch[0]["esmc_features"].shape[1]
    
    # 创建输出批处理字典
    collated = {
        "sample_id": [],
        "esm2_features": torch.zeros((batch_size, max_len, esm2_dim)),
        "esmc_features": torch.zeros((batch_size, max_len, esmc_dim)),
        "mask": torch.zeros((batch_size, max_len), dtype=torch.bool)
    }
    
    # 填充批处理数据
    for i, item in enumerate(batch):
        seq_len = item["esm2_features"].shape[0]
        
        collated["sample_id"].append(item["sample_id"])
        collated["esm2_features"][i, :seq_len] = torch.tensor(item["esm2_features"])
        collated["esmc_features"][i, :seq_len] = torch.tensor(item["esmc_features"])
        collated["mask"][i, :seq_len] = torch.tensor(item["mask"])
        
    return collated

def extract_features(sequence: str, output_dir: str = "./features", model_type: str = "both"):
    """提取蛋白质序列特征
    
    Args:
        sequence: 蛋白质序列
        output_dir: 输出目录
        model_type: 要使用的模型类型: "esm2", "esmc", 或 "both"
        
    Returns:
        Dict: 包含生成的特征文件路径和样本ID的字典
    """
    try:
        # 导入特征提取库
        from protein_feature_extractor import extract_features as extract_model_features
        
        os.makedirs(output_dir, exist_ok=True)
        esm2_dir = os.path.join(output_dir, "esm2_features")
        esmc_dir = os.path.join(output_dir, "esmc_features")
        os.makedirs(esm2_dir, exist_ok=True)
        os.makedirs(esmc_dir, exist_ok=True)
        
        # 使用统一的样本ID生成方式
        sample_id = f"sample_{hash(sequence) % 10000:04d}"
        result = {"sample_id": sample_id, "sequence": sequence}
        
        if model_type in ["esm2", "both"]:
            # 提取ESM2特征
            esm2_features = extract_model_features(sequence, "esm2")
            # 确保特征包含sample_id
            if "sample_id" not in esm2_features:
                esm2_features["sample_id"] = sample_id
            esm2_file = os.path.join(esm2_dir, f"{sample_id}_features.npy")
            np.save(esm2_file, esm2_features)
            result["esm2_file"] = esm2_file
            
        if model_type in ["esmc", "both"]:
            # 提取ESMC特征
            esmc_features = extract_model_features(sequence, "esmc")
            # 确保特征包含sample_id
            if "sample_id" not in esmc_features:
                esmc_features["sample_id"] = sample_id
            esmc_file = os.path.join(esmc_dir, f"{sample_id}_features.npy")
            np.save(esmc_file, esmc_features)
            result["esmc_file"] = esmc_file
            
        return result  # 返回包含样本ID和文件路径的结果
        
    except ImportError:
        logging.error("无法导入特征提取库，请确保protein_feature_extractor已安装")
        raise
    except Exception as e:
        logging.error(f"特征提取失败: {str(e)}")
        raise