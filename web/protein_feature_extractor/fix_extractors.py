import logging
import torch
from .feature_manager import FeatureManager
from .esm2_extractor import ESM2Extractor
from .esmc_extractor import ESMCExtractor

logger = logging.getLogger(__name__)

# 全局缓存，保存已加载的提取器实例
_EXTRACTORS_CACHE = {}

def get_extractor(model_type="esm2", model_path=None, device=None, force_reload=False):
    """获取或创建特征提取器，避免重复加载

    Args:
        model_type: 模型类型 "esm2" 或 "esmc"
        model_path: 可选的模型路径
        device: 计算设备
        force_reload: 是否强制重新加载模型

    Returns:
        配置好的特征提取器实例
    """
    global _EXTRACTORS_CACHE
    
    # 默认模型路径
    default_paths = {
        "esm2": "facebook/esm2_t33_650M_UR50D",
        "esmc": "esmc_600m"
    }
    
    model_type = model_type.lower()
    cache_key = f"{model_type}_{model_path or default_paths[model_type]}"
    
    # 如果缓存中已存在且不需要强制重载，则返回缓存的实例
    if not force_reload and cache_key in _EXTRACTORS_CACHE:
        logger.info(f"使用缓存的{model_type}提取器")
        return _EXTRACTORS_CACHE[cache_key]
    
    # 配置
    config = {
        "model_name": model_type,
        "model_path": model_path if model_path else default_paths[model_type],
        "enabled": True,
        "cache_dir": None  # 可以添加缓存目录配置
    }
    
    # 创建提取器
    if model_type == "esm2":
        extractor = ESM2Extractor(config, device)
    elif model_type == "esmc":
        extractor = ESMCExtractor(config, device)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 保存到缓存
    _EXTRACTORS_CACHE[cache_key] = extractor
    
    return extractor

def extract_features_robust(sequence, model_type="esm2", model_path=None, max_len=1024, device=None):
    """增强版特征提取，更健壮地处理模型加载和重用
    
    Args:
        sequence: 蛋白质序列
        model_type: 模型类型
        model_path: 可选的模型路径
        max_len: 最大序列长度
        device: 计算设备
        
    Returns:
        提取的特征字典
    """
    try:
        # 生成样本ID
        sample_id = f"sample_{hash(sequence) % 10000:04d}"
        
        # 获取或创建提取器
        extractor = get_extractor(model_type, model_path, device)
        
        # 提取特征
        residue_repr, mask, global_repr = extractor.extract_features(sequence, max_len)
        
        return {
            "residue_representation": residue_repr,
            "mask": mask,
            "global_representation": global_repr,
            "sample_id": sample_id,  # 添加样本ID
            "sequence": sequence     # 添加原始序列
        }
    except Exception as e:
        logger.error(f"特征提取失败: {str(e)}")
        # 尝试重新加载提取器
        logger.info(f"尝试重新加载{model_type}提取器...")
        try:
            extractor = get_extractor(model_type, model_path, device, force_reload=True)
            residue_repr, mask, global_repr = extractor.extract_features(sequence, max_len)
            
            # 生成样本ID（在异常处理中也需要）
            sample_id = f"sample_{hash(sequence) % 10000:04d}"
            
            return {
                "residue_representation": residue_repr,
                "mask": mask,
                "global_representation": global_repr,
                "sample_id": sample_id,  # 添加样本ID
                "sequence": sequence     # 添加原始序列
            }
        except Exception as e2:
            logger.error(f"重新加载后提取仍然失败: {str(e2)}")
            raise