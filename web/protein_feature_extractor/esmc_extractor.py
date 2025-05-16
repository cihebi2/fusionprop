import torch
import logging
# import multiprocessing as mp # Lock is now handled by ManagedModel
from typing import Tuple, Dict, Any, Optional
from .base_extractor import FeatureExtractor
from .model_lifecycle import ManagedModel # Added import

class ESMCExtractor(FeatureExtractor):
    """ESM-C特征提取器"""
    def __init__(self, config: Dict[str, Any], device: Optional[torch.device]): # Reverted: managed_model removed
        super().__init__(config, device)
        self.logger = logging.getLogger("ESMCExtractor")
        self.managed_model: Optional[ManagedModel] = None # Add optional attribute
        self._model_loaded_internally = False
    
    def _load_model_impl(self) -> None:
        """加载ESM-C模型 (如果不由ManagedModel管理)"""
        if self.managed_model:
            # self.logger.info("ESMC model is managed by ManagedModel. Skipping internal load.")
            return
            
        self.logger.info(f"Internally loading ESMC model as not managed: {self.config.get('model_path')}")
        # Reinstating original loading logic as fallback
        # Ensure the correct esm library (THUDM version) is installed
        torch.cuda.empty_cache() # Clear cache before attempting load
        try:
            from esm import pretrained # Or specific imports like ESMC
            model_path = self.config['model_path']
            
            # --- Critical Part: Replicate original loading --- 
            # This needs to match how ESMC model should be loaded based on model_path
            # (e.g., local vs Hub ID) and the specific functions available in the installed esm library
            if "load_local_model" in dir(pretrained): # Check for the specific function
                 self.logger.info(f"Attempting to load ESMC locally from: {model_path} using load_local_model")
                 # model_obj, _ = pretrained.load_local_model(model_path) 
                 self.model = pretrained.load_local_model(model_path) # Assume it returns model directly
                 self.logger.info(f"Successfully called load_local_model for {model_path}")
            else:
                 # Fallback assumption: maybe model_path is Hub ID compatible with another func?
                 # This part is uncertain without knowing the exact esm library API in use.
                 self.logger.warning(f"'load_local_model' not found. Attempting fallback load for ESMC: {model_path}. This might fail or be incorrect.")
                 # Example: Using from_pretrained from the ESMC class if available
                 try:
                     from esm.models.esmc import ESMC
                     self.model = ESMC.from_pretrained(model_path)
                 except ImportError:
                     self.logger.error("Failed to import ESMC class. Cannot use from_pretrained.")
                     raise # Re-raise import error if ESMC class isn't found
                 except Exception as e_load:
                     self.logger.error(f"Fallback load using ESMC.from_pretrained failed: {e_load}")
                     raise # Re-raise other loading errors
            
            # --------------------------------------------------
            
            if self.model:
                self.model = self.model.to(self.device).eval()
                self._model_loaded_internally = True
                self.logger.info(f"ESMC model loaded internally. Device: {self.device}")
            else:
                 raise RuntimeError("Internal ESMC loading resulted in None model.")
                 
        except ImportError as e_imp:
            self.logger.error(f"Internal load failed: Cannot import from esm library. Install THUDM version? Error: {e_imp}", exc_info=True)
            self.model = None
        except RuntimeError as e_rt:
            if "CUDA out of memory" in str(e_rt):
                self.logger.warning(f"Internal load failed: CUDA out of memory. Try reducing usage or using CPU. Error: {e_rt}")
            else:
                self.logger.error(f"Internal load failed: Runtime error. Error: {e_rt}", exc_info=True)
            self.model = None
        except Exception as e_gen:
             self.logger.error(f"Internal load failed: General error. Error: {e_gen}", exc_info=True)
             self.model = None
    
    def extract_features(self, sequence: str, max_len: int = 1024) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model_to_use = None
        if self.managed_model:
            model_to_use = self.managed_model.get()
        else:
            if self.model is None:
                 self.load_model()
            model_to_use = self.model

        if model_to_use is None:
            self.logger.error("ESM-C model not available.")
            raise RuntimeError("ESM-C模型未加载成功，无法提取特征")
            
        with torch.no_grad():
            try:
                current_model_device = next(model_to_use.parameters()).device
                try:
                    from esm.sdk.api import ESMProtein, LogitsConfig
                except ImportError as e:
                    self.logger.error(f"Failed to import from esm.sdk.api: {e}. Ensure THUDM ESMC library is correctly installed and structured.")
                    raise
                
                protein = ESMProtein(sequence=sequence)
                protein_tensor = model_to_use.encode(protein)
                if protein_tensor.device != current_model_device:
                    protein_tensor = protein_tensor.to(current_model_device)
                
                logits_output = model_to_use.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True)
                )
                embeddings = logits_output.embeddings[0][1:-1].cpu() 
                current_len = embeddings.shape[0]
                target_len = min(max_len if max_len is not None else 1022, 1022)

                if current_len == 0:
                    output_dim = model_to_use.args.embed_dim if hasattr(model_to_use, 'args') and hasattr(model_to_use.args, 'embed_dim') else 1280
                    padded_residue = torch.zeros((target_len, output_dim))
                    padded_mask = torch.zeros(target_len, dtype=torch.bool)
                    global_representation = torch.zeros(output_dim)
                else:
                    if current_len < target_len:
                        pad_amount = target_len - current_len
                        padded_residue = torch.nn.functional.pad(embeddings, (0, 0, 0, pad_amount), "constant", 0)
                        padded_mask = torch.cat((torch.ones(current_len, dtype=torch.bool), torch.zeros(pad_amount, dtype=torch.bool)))
                    elif current_len > target_len:
                        padded_residue = embeddings[:target_len]
                        padded_mask = torch.ones(target_len, dtype=torch.bool)
                    else:
                        padded_residue = embeddings
                        padded_mask = torch.ones(target_len, dtype=torch.bool)
                    global_representation = embeddings.mean(dim=0)
                
                return padded_residue, padded_mask, global_representation
                    
            except Exception as e:
                self.logger.error(f"ESM-C特征提取失败: {e}", exc_info=True)
                raise