"""
蛋白质特性预测共享数据处理模块
"""
import os
import torch
import numpy as np
import logging
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    """特征数据集，支持基本特征提取和可选的归一化"""
    def __init__(self, esm2_dir, esmc_dir, config, sample_ids=None, normalization_method=None):
        self.esm2_dir = esm2_dir
        self.esmc_dir = esmc_dir
        self.config = config
        self.normalization_method = normalization_method or getattr(config, 'normalization_method', None)
        
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
        try:
            sample_id = self.common_ids[idx]
            
            # 加载特征
            esm2_data = np.load(os.path.join(self.esm2_dir, self.esm2_files[sample_id]), allow_pickle=True).item()
            esmc_data = np.load(os.path.join(self.esmc_dir, self.esmc_files[sample_id]), allow_pickle=True).item()
            
            # 获取特征并确保是正确的类型
            if isinstance(esm2_data["residue_representation"], torch.Tensor):
                esm2_features = esm2_data["residue_representation"].float()
            else:
                esm2_features = torch.tensor(esm2_data["residue_representation"], dtype=torch.float32)
                
            if isinstance(esmc_data["residue_representation"], torch.Tensor):
                esmc_features = esmc_data["residue_representation"].float()
            else:
                esmc_features = torch.tensor(esmc_data["residue_representation"], dtype=torch.float32)
            
            # 维度规范化
            if esm2_features.dim() > 2:
                esm2_features = esm2_features.squeeze(0)
            if esmc_features.dim() > 2:
                esmc_features = esmc_features.squeeze(0)
            
            # 获取掩码
            mask = esm2_data["mask"]
            if isinstance(mask, torch.Tensor):
                mask = mask.bool()
            else:
                mask = torch.tensor(mask, dtype=torch.bool)
                
            if mask.dim() > 1:
                mask = mask.squeeze(0)
            
            # 特征归一化
            if self.normalization_method:
                esm2_features = self._normalize_features(esm2_features, "esm2")
                esmc_features = self._normalize_features(esmc_features, "esmc")
            else:
                # 使用基本标准化
                esm2_features = (esm2_features - self.config.esm2_mean) / self.config.esm2_std
                esmc_features = (esmc_features - self.config.esmc_mean) / self.config.esmc_std
            
            # 准备基本结果字典
            result = {
                "sample_id": sample_id,
                "esm2_features": esm2_features,
                "esmc_features": esmc_features,
                "mask": mask
            }
            
            # 检查是否有其他标签（如溶解度、毒性等）
            for label_name in ["solubility", "toxicity", "thermostability"]:
                if label_name in esm2_data:
                    result[label_name] = torch.tensor(esm2_data[label_name], dtype=torch.float32)
            
            return result
            
        except Exception as e:
            logging.error(f"加载样本 {self.common_ids[idx]} 出错: {str(e)}")
            # 返回一个小尺寸的dummy样本
            return {
                "sample_id": self.common_ids[idx],
                "esm2_features": torch.zeros(10, getattr(self.config, 'esm2_dim', 1280)),
                "esmc_features": torch.zeros(10, getattr(self.config, 'esmc_dim', 1152)),
                "mask": torch.zeros(10, dtype=torch.bool)
            }
    
    def _normalize_features(self, features, feature_type="esm2"):
        """根据配置的方法归一化特征"""
        # 基于方法选择归一化策略
        if self.normalization_method == "global":
            # 使用全局均值和标准差
            if feature_type == "esm2":
                return (features - self.config.esm2_mean) / self.config.esm2_std
            else:  # esmc
                return (features - self.config.esmc_mean) / self.config.esmc_std
                
        elif self.normalization_method == "sequence":
            # 对每个序列单独归一化
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True) + 1e-6
            return (features - mean) / std
            
        # 默认情况：使用全局标准化
        if feature_type == "esm2":
            return (features - self.config.esm2_mean) / self.config.esm2_std
        else:  # esmc
            return (features - self.config.esmc_mean) / self.config.esmc_std

def feature_collate_fn(batch):
    """特征批处理函数，支持可选的标签字段"""
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
    
    # 检查是否有其他标签字段
    extra_fields = []
    for field in batch[0].keys():
        if field not in ["sample_id", "esm2_features", "esmc_features", "mask"]:
            if isinstance(batch[0][field], torch.Tensor) and batch[0][field].dim() == 0:
                collated[field] = []
                extra_fields.append(field)
    
    # 填充批处理数据
    for i, item in enumerate(batch):
        seq_len = item["esm2_features"].shape[0]
        
        collated["sample_id"].append(item["sample_id"])
        collated["esm2_features"][i, :seq_len] = item["esm2_features"]
        collated["esmc_features"][i, :seq_len] = item["esmc_features"]
        collated["mask"][i, :seq_len] = item["mask"]
        
        # 添加额外字段
        for field in extra_fields:
            if field in item:
                collated[field].append(item[field])
    
    # 将额外标签字段转换为张量
    for field in extra_fields:
        if collated[field]:
            try:
                collated[field] = torch.stack(collated[field])
            except:
                pass
    
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
            # 如果特征是tensor，转换为numpy
            for key in esm2_features:
                if isinstance(esm2_features[key], torch.Tensor):
                    esm2_features[key] = esm2_features[key].detach().cpu().numpy()
            # 确保特征包含sample_id
            if "sample_id" not in esm2_features:
                esm2_features["sample_id"] = sample_id
            esm2_file = os.path.join(esm2_dir, f"{sample_id}_features.npy")
            np.save(esm2_file, esm2_features)
            result["esm2_file"] = esm2_file
            
        if model_type in ["esmc", "both"]:
            # 提取ESMC特征
            esmc_features = extract_model_features(sequence, "esmc")
            # 如果特征是tensor，转换为numpy
            for key in esmc_features:
                if isinstance(esmc_features[key], torch.Tensor):
                    esmc_features[key] = esmc_features[key].detach().cpu().numpy()
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

# 为兼容性保留原来溶解性预测中的类名
NormalizedDataset = FeatureDataset
normalized_collate_fn = feature_collate_fn