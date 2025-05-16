"""
蛋白质溶解性预测子包

提供蛋白质序列的溶解性预测功能。
"""

# 导入主要函数，使其可以直接从子包中访问
from .predictor import predict_solubility, batch_predict