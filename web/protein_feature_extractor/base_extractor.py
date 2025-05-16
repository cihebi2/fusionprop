import torch
import os
import logging
from typing import Tuple, Dict, Any, Optional, Union

class FeatureExtractor:
    """特征提取器基类"""
    _model_registry = {}  # 全局模型注册表，用于跨实例共享
    
    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None):
        """
        初始化特征提取器
        
        Args:
            config: 配置字典，包含模型路径等信息
            device: 指定运行设备，None则自动选择
        """
        self.config = config
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._model_key = f"{self.__class__.__name__}_{config.get('model_name', 'unknown')}"
        self.logger = logging.getLogger("FeatureExtractor")
    
    def load_model(self) -> Any:
        """加载模型，优先从注册表获取"""
        if self._model_key in self._model_registry:
            self.logger.info(f"从注册表获取已加载的{self.config.get('model_name', 'unknown')}模型")
            self.model = self._model_registry[self._model_key]
            return self.model
            
        # 加载新模型的逻辑（子类实现）
        self._load_model_impl()
        
        # 加载成功后注册到全局
        if self.model is not None:
            self._model_registry[self._model_key] = self.model
        return self.model
    
    def _load_model_impl(self) -> None:
        """实际加载模型的实现（子类必须重写）"""
        raise NotImplementedError("子类必须实现此方法")
    
    def extract_features(self, sequence: str, max_len: int = 1024) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        提取特征
        
        Args:
            sequence: 蛋白质序列
            max_len: 最大序列长度
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (residue_representation, mask, global_representation)
        """
        if self.model is None:
            self.load_model()
        
        # 特征提取逻辑（子类实现）
        raise NotImplementedError("子类必须实现此方法")
    
    def cleanup(self) -> None:
        """清理资源（只在程序结束时调用）"""
        # 从注册表中移除
        if self._model_key in self._model_registry:
            del self._model_registry[self._model_key]
            self.logger.info(f"已从注册表移除{self.config.get('model_name', 'unknown')}模型")
        
        # 清理内存
        self.model = None
        torch.cuda.empty_cache()