import logging
from typing import Dict, Any, Optional
import torch
from .feature_manager import FeatureManager
from .esm2_extractor import ESM2Extractor
from .esmc_extractor import ESMCExtractor
from .base_extractor import FeatureExtractor
from .fix_extractors import extract_features_robust
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

__all__ = ['FeatureManager', 'ESM2Extractor', 'ESMCExtractor', 'FeatureExtractor', 'extract_features']


# 修改提取函数，使用健壮版特征提取
def extract_features(sequence: str, model_type: str = "esm2", 
                   model_path: Optional[str] = None, 
                   max_len: int = 1024, 
                   device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    便捷函数，一步提取蛋白质序列特征
    
    Args:
        sequence: 蛋白质序列
        model_type: 模型类型，支持 "esm2" 或 "esmc"
        model_path: 模型路径，如果为None则使用默认路径
        max_len: 最大序列长度
        device: 计算设备
        
    Returns:
        Dict: 包含residue_representation和mask的字典
    """
    # 使用健壮版提取功能
    return extract_features_robust(sequence, model_type, model_path, max_len, device)