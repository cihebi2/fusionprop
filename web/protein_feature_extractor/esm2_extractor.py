import torch
import logging
from typing import Tuple, Dict, Any, Optional
# Transformers are loaded by FeatureManager's load function now
# from transformers import AutoTokenizer, AutoModel, AutoConfig 
from .base_extractor import FeatureExtractor
from .model_lifecycle import ManagedModel # Added import

class ESM2Extractor(FeatureExtractor):
    """ESM2特征提取器"""
    def __init__(self, config: Dict[str, Any], device: Optional[torch.device]): # Reverted: managed_model removed from __init__ args
        super().__init__(config, device)
        self.logger = logging.getLogger("ESM2Extractor")
        self.managed_model: Optional[ManagedModel] = None # Add as an optional attribute
        self.tokenizer = None # Tokenizer will be loaded by the original _load_model_impl or set if managed
        self._model_loaded_internally = False

    def _load_model_impl(self) -> None:
        """加载ESM2模型和tokenizer (如果不由ManagedModel管理)"""
        # This method is called by self.load_model() from base_extractor if model is None
        if self.managed_model: # If managed, model access is through managed_model.get()
            # self.logger.info("ESM2 model is managed by ManagedModel. Skipping internal load.")
            # The actual model and tokenizer will be fetched in extract_features
            return

        self.logger.info(f"Internally loading ESM2 model and tokenizer as not managed: {self.config.get('model_path')}")
        from transformers import AutoTokenizer, AutoModel, AutoConfig
        try:
            model_path = self.config['model_path']
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            model_config_transformers = AutoConfig.from_pretrained(model_path, output_hidden_states=True)
            model_config_transformers.hidden_dropout = 0.
            model_config_transformers.hidden_dropout_prob = 0.
            model_config_transformers.attention_dropout = 0.
            model_config_transformers.attention_probs_dropout_prob = 0.
            
            self.model = AutoModel.from_pretrained(model_path, config=model_config_transformers)
            self.model = self.model.to(self.device).eval()
            self._model_loaded_internally = True
            self.logger.info(f"ESM2 model and tokenizer loaded internally. Device: {self.device}")
        except Exception as e:
            self.logger.error(f"Internal ESM2 model loading failed: {e}", exc_info=True)
            self.model = None
            self.tokenizer = None
    
    def extract_features(self, sequence: str, max_len: int = 1024) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model_to_use = None
        tokenizer_to_use = None

        if self.managed_model:
            # Get model and tokenizer from ManagedModel
            loaded_obj = self.managed_model.get()
            if isinstance(loaded_obj, tuple) and len(loaded_obj) == 2:
                model_to_use, tokenizer_to_use = loaded_obj
            else:
                model_to_use = loaded_obj # Assuming it might just return model if old load_function is used
                self.logger.warning("ManagedModel for ESM2 did not return a (model, tokenizer) tuple. Assuming old load function. Tokenizer might be missing or old.")
                # Attempt to use self.tokenizer if it was set by an internal load or pre-set
                if not tokenizer_to_use and self.tokenizer:
                    tokenizer_to_use = self.tokenizer
                elif not tokenizer_to_use:
                    # Last resort: try to load tokenizer here if model is present
                    # This is not ideal and indicates a setup issue with ManagedModel's load_function
                    try:
                        model_path_from_config = self.config.get('model_path', 'facebook/esm2_t33_650M_UR50D')
                        tokenizer_to_use = AutoTokenizer.from_pretrained(model_path_from_config)
                        self.tokenizer = tokenizer_to_use # Cache it
                        self.logger.info(f"Fallback: Loaded tokenizer for ESM2 on-the-fly in extract_features.")
                    except Exception as e_tok:
                        self.logger.error(f"Fallback: Failed to load tokenizer for ESM2 in extract_features: {e_tok}")

        else: # Not managed, use internal model and tokenizer
            if self.model is None or self.tokenizer is None:
                self.load_model() # This will call _load_model_impl
            model_to_use = self.model
            tokenizer_to_use = self.tokenizer

        if model_to_use is None or tokenizer_to_use is None:
            self.logger.error("ESM2 model or tokenizer not available.")
            raise RuntimeError("ESM2 model or tokenizer failed to load.")
            
        with torch.no_grad():
            spaced_seq = " ".join(list(sequence))
            inputs = tokenizer_to_use.encode_plus(
                spaced_seq, 
                return_tensors="pt", 
                add_special_tokens=True,
                padding="max_length",
                max_length=min(len(sequence.replace(" ", "")) + 2, max_len if max_len is not None else 10240),
                truncation=True
            )
            inputs_on_device = {k: v.to(model_to_use.device) for k, v in inputs.items()}
            outputs = model_to_use(**inputs_on_device)
            last_hidden_states = outputs.last_hidden_state
            attention_mask = inputs_on_device['attention_mask'][0].bool()
            tokens_under_mask = last_hidden_states[0, attention_mask]
            
            if tokens_under_mask.size(0) < 2:
                output_dim = model_to_use.config.hidden_size 
                encoded_seq = torch.zeros((0, output_dim), device=model_to_use.device)
            else:
                encoded_seq = tokens_under_mask[1:-1]
            
            encoded_seq = encoded_seq.cpu()
            current_len = encoded_seq.shape[0]
            target_len = min(max_len if max_len is not None else 1022, 1022)

            if current_len < target_len:
                pad_amount = target_len - current_len
                padded_residue = torch.nn.functional.pad(encoded_seq, (0, 0, 0, pad_amount), "constant", 0)
                padded_mask = torch.cat((torch.ones(current_len, dtype=torch.bool), torch.zeros(pad_amount, dtype=torch.bool)))
            elif current_len > target_len:
                padded_residue = encoded_seq[:target_len]
                padded_mask = torch.ones(target_len, dtype=torch.bool)
            else:
                padded_residue = encoded_seq
                padded_mask = torch.ones(target_len, dtype=torch.bool)
            
            if current_len > 0:
                 global_representation = encoded_seq.mean(dim=0)
            else:
                 global_representation = torch.zeros(padded_residue.size(1) if padded_residue.numel() > 0 else model_to_use.config.hidden_size) 
            
            return padded_residue, padded_mask, global_representation