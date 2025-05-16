import torch
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from .esm2_extractor import ESM2Extractor
from .esmc_extractor import ESMCExtractor
from .model_lifecycle import ManagedModel

class FeatureManager:
    """特征管理器，用于管理多种特征提取器，并自动管理模型生命周期"""
    def __init__(self, configs: Dict[str, Dict[str, Any]], device: Optional[torch.device] = None, model_release_timeout: int = 10):
        """
        初始化特征管理器
        
        Args:
            configs: 模型配置字典，格式: {"esm2": {"model_path": "...", ...}, "esmc": {...}}
            device: 计算设备，默认为自动选择
            model_release_timeout: 模型不活动多少秒后释放（默认600秒=10分钟）
        """
        self.configs = configs
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_release_timeout = model_release_timeout
        self.extractors: Dict[str, Union[ESM2Extractor, ESMCExtractor]] = {}
        self.managed_models: Dict[str, ManagedModel] = {}
        self.logger = logging.getLogger("FeatureManager")
        
        self.logger.info(f"FeatureManager initialized. Device: {self.device}, Model Timeout: {self.model_release_timeout}s")
        
        # 注册已启用的模型配置
        for name, model_config in self.configs.items():
            if model_config.get("enabled", True):
                self.register_extractor(name, model_config)
            else:
                 self.logger.info(f"Skipping disabled extractor: {name}")
    
    def _load_esm2_model_internal(self, config: Dict[str, Any]):
        """Internal function to load the actual ESM2 model and tokenizer."""
        self.logger.info(f"Executing ESM2 load function (path: {config.get('model_path')})...")
        from transformers import AutoModel, AutoTokenizer, AutoConfig # Ensure AutoModel is the correct one
        
        model_path = config.get('model_path', 'facebook/esm2_t33_650M_UR50D')
        self.logger.info(f"Loading ESM2 model and tokenizer from: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model_config_transformers = AutoConfig.from_pretrained(model_path, output_hidden_states=True)
        model_config_transformers.hidden_dropout = 0.
        model_config_transformers.hidden_dropout_prob = 0.
        model_config_transformers.attention_dropout = 0.
        model_config_transformers.attention_probs_dropout_prob = 0.
        
        model = AutoModel.from_pretrained(model_path, config=model_config_transformers)
        model = model.to(self.device).eval()
        
        self.logger.info("ESM2 model and tokenizer loaded via internal function.")
        return model, tokenizer # Return both model and tokenizer

    def _load_esmc_model_internal(self, config: Dict[str, Any]):
        model_path = config.get('model_path')
        self.logger.info(f"Loading ESMC from: {model_path}")
        if not model_path:
            self.logger.error("ESMC model path not configured.")
            return None
        try:
            from esm import pretrained # Assuming this is the THUDM version
            # model, alphabet = pretrained.load_local_model(model_path) # Original, caused error
            model = pretrained.load_local_model(model_path) # Corrected: load_local_model returns model directly
            # The alphabet for ESMC is usually handled internally or via esm.data.Alphabet if needed by tokenizer
            # For ESMC from THUDM, the model object itself is typically used with its own methods.
            # We don't return alphabet here; ESMCExtractor will use the model object.
            self.logger.info(f"Successfully loaded ESMC model from {model_path}")
            return model # Return only the model
        except TypeError as te:
            if "cannot unpack non-iterable ESMC object" in str(te):
                 self.logger.error(f"Failed to load ESMC model from {model_path}: {te} - This indicates load_local_model did not return a tuple as expected by previous code, but returned the model itself.", exc_info=True)
            else:
                 self.logger.error(f"Failed to load ESMC model from {model_path} due to TypeError: {te}", exc_info=True)
            raise # Re-raise the error to be caught by ManagedModel
        
    def register_extractor(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Registers a feature extractor with automatic model lifecycle management.
        
        Args:
            name: Extractor name (e.g., "esm2", "esmc")
            config: Extractor configuration
            
        Returns:
            bool: True if registration was successful, False otherwise.
        """
        if not config.get("enabled", True):
            self.logger.info(f"Feature extractor {name} is disabled in config.")
            return False
            
        if name in self.extractors:
            self.logger.warning(f"Extractor {name} already registered. Skipping.")
            return True

        try:
            managed_model_instance = None
            extractor_instance = None
            
            if name.lower() == "esm2":
                load_func = lambda cfg=config: self._load_esm2_model_internal(cfg)
                managed_model_instance = ManagedModel(
                    load_function=load_func, 
                    model_name=f"ESM2 ({config.get('model_path', 'default')})", 
                    release_timeout_seconds=self.model_release_timeout
                )
                extractor_instance = ESM2Extractor(config, self.device)
                extractor_instance.managed_model = managed_model_instance

            elif name.lower() == "esmc":
                load_func = lambda cfg=config: self._load_esmc_model_internal(cfg)
                managed_model_instance = ManagedModel(
                    load_function=load_func, 
                    model_name=f"ESMC ({config.get('model_path', 'default')})", 
                    release_timeout_seconds=self.model_release_timeout
                )
                extractor_instance = ESMCExtractor(config, self.device)
                extractor_instance.managed_model = managed_model_instance
                
            else:
                self.logger.warning(f"Attempted to register unknown extractor type: {name}")
                return False

            self.managed_models[name] = managed_model_instance
            self.extractors[name] = extractor_instance
            
            self.logger.info(f"Successfully registered extractor: {name} with ManagedModel.")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register extractor {name}: {e}", exc_info=True)
            if name in self.managed_models: del self.managed_models[name]
            if name in self.extractors: del self.extractors[name]
            return False
    
    def extract_features(self, sequence: str, model_name: str, max_len: int = 1024) -> Dict[str, torch.Tensor]:
        """
        使用指定模型提取序列特征. 
        模型将通过 ManagedModel 自动加载/获取。
        
        Args:
            sequence: 蛋白质序列
            model_name: 模型名称 ("esm2", "esmc")
            max_len: 最大序列长度
            
        Returns:
            Dict: 包含residue_representation, mask, global_representation 的字典
        """
        if model_name not in self.extractors:
             self.logger.error(f"Extractor {model_name} not found. Available: {list(self.extractors.keys())}")
             raise ValueError(f"Model extractor '{model_name}' is not registered or enabled.")
            
        extractor = self.extractors[model_name]
        
        try:
            residue_repr, mask, global_repr = extractor.extract_features(sequence, max_len)
            
            return {
                "residue_representation": residue_repr,
                "mask": mask,
                "global_representation": global_repr
            }
        except Exception as e:
            self.logger.error(f"Feature extraction failed for model '{model_name}' on sequence '{sequence[:30]}...': {e}", exc_info=True)
            raise
    
    def extract_all_features(self, sequence: str, max_len: int = 1024) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        使用所有已注册模型提取序列特征
        
        Args:
            sequence: 蛋白质序列
            max_len: 最大序列长度
            
        Returns:
            Dict: 包含各模型特征的字典 {"model_name": {"residue_representation": tensor, "mask": tensor, ...}}
        """
        all_features = {}
        for name in self.extractors.keys(): 
            try:
                features = self.extract_features(sequence, name, max_len)
                all_features[name] = features
            except Exception as e:
                self.logger.warning(f"Failed to extract features using {name} for sequence '{sequence[:30]}...'. Returning zeros. Error: {e}")
                output_dim = self.configs.get(name, {}).get("output_dim", 1280)
                seq_length = min(len(sequence), max_len)
                all_features[name] = {
                    "residue_representation": torch.zeros(seq_length, output_dim, device=self.device),
                    "mask": torch.zeros(seq_length, dtype=torch.bool, device=self.device),
                    "global_representation": torch.zeros(output_dim, device=self.device)
                }
        
        return all_features
    
    def shutdown(self):
        """Explicitly releases all managed models and stops checker threads."""
        self.logger.info("Shutting down FeatureManager, releasing all models...")
        for name, managed_model in self.managed_models.items():
            try:
                del managed_model 
            except Exception as e:
                self.logger.error(f"Error during shutdown releasing model {name}: {e}", exc_info=True)
        self.managed_models.clear()
        self.extractors.clear()
        try:
             if torch.cuda.is_available():
                   torch.cuda.empty_cache()
        except:
            pass
        self.logger.info("FeatureManager shutdown complete.")

    def __del__(self):
        self.shutdown()