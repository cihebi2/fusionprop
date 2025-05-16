import logging

logger = logging.getLogger(__name__)

def get_default_feature_configs():
    """
    Returns the default configurations for feature extractors (ESM2, ESMC).
    These are used by FeatureManager if no specific configs are provided.
    """
    # TODO: Consider moving these paths to Django settings or a more robust config file
    # For now, using common Hugging Face IDs or expected local names.
    # Ensure these model paths are correct for your setup (downloaded locally or HF Hub ID).
    
    # Default ESM2 model - facebook/esm2_t33_650M_UR50D is a common one
    # default_esm2_path = "esm2_t33_650M_UR50D" # Or your specific local path / HF ID
    default_esm2_path = "facebook/esm2_t33_650M_UR50D" # Using full Hugging Face ID

    # Default ESMC model - e.g., 'esmc_600m' if that's your local directory name or a known ID
    # Or, if it's a direct path like from original config: "/path/to/your/esmc_models/esmc_600m"
    default_esmc_path = "esmc_600m" # Or your specific local path / HF ID

    logger.info(f"Providing default feature configs: ESM2_PATH='{default_esm2_path}', ESMC_PATH='{default_esmc_path}'")

    configs = {
        "esm2": {
            "model_path": default_esm2_path, 
            "enabled": True,
            "name": "ESM-2 (Default)",
            # Add other necessary parameters if your ESM2Extractor or FeatureManager expects them
            # e.g., "max_length": 1022, "device": "cuda" (device is handled by FeatureManager itself)
        },
        "esmc": {
            "model_path": default_esmc_path, 
            "enabled": True,
            "name": "ESM-C (Default)",
            # Add other necessary parameters
        }
        # You could add other feature extractors here if FeatureManager supports them
    }
    return configs

# Example of how you might load from Django settings if preferred:
# from django.conf import settings
# def get_feature_configs_from_settings():
#     return settings.FEATURE_EXTRACTOR_CONFIGS 