"""
蛋白质预测共享组件
"""
from .utils import set_weights_only_warning, setup_logger
from .data import FeatureDataset, feature_collate_fn, extract_features