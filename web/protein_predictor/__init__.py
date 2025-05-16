"""
蛋白质特性预测库 - 预测蛋白质的各种特性，包括毒性、热稳定性和溶解性

示例用法:
    # 单特性预测
    from protein_predictor import predict_toxicity
    result = predict_toxicity("蛋白质序列", return_confidence=True)
    
    # 同时预测所有特性 (单序列)
    from protein_predictor import predict_all
    result = predict_all("蛋白质序列", return_confidence=True)
    
    # 批量预测 (高效方式)
    from protein_predictor import ProteinPredictorManager
    manager = ProteinPredictorManager()
    results = manager.batch_predict_all(sequences, names)
"""
from pathlib import Path

# 导入主要函数，使其可以直接从包中访问
from .toxicity.predictor import predict_toxicity, batch_predict as batch_predict_toxicity
from .thermostability.predictor import predict_thermostability, batch_predict as batch_predict_thermostability
from .solubility.predictor import predict_solubility, batch_predict as batch_predict_solubility
from .common.data import extract_features
from .predictor_manager import ProteinPredictorManager

# 保持原有的predict_all函数，但在文档中建议批量任务使用ProteinPredictorManager
from .predictor_manager import ProteinPredictorManager

# 导出版本信息
__version__ = "1.0.0"

# 为了保持向后兼容性，保留原始的predict_all函数
def predict_all(sequence: str, return_confidence: bool = False, 
                toxicity_model_dir: str = None, 
                thermostability_model_dir: str = None,
                solubility_model_dir: str = None,
                cleanup_features: bool = True) -> dict:
    """
    同时预测蛋白质的毒性、热稳定性和溶解性，共享特征提取过程
    注意：如果需要批量预测，建议使用ProteinPredictorManager以获得更好的性能
    """
    # 使用临时预测管理器进行单序列预测
    manager = ProteinPredictorManager(
        toxicity_model_dir=toxicity_model_dir,
        thermostability_model_dir=thermostability_model_dir,
        solubility_model_dir=solubility_model_dir
    )
    return manager.predict_all(sequence, return_confidence, cleanup_features)

# 为了保持向后兼容性，保留原始的batch_predict_all函数，但内部使用管理器实现
def batch_predict_all(sequences, names=None, with_confidence=True, export_csv=None):
    """
    批量同时预测多个蛋白质序列的所有特性
    注意：此函数每次都会重新加载模型，如果批处理较大，建议直接使用ProteinPredictorManager
    """
    # 提醒用户使用更高效的方式
    print("注意：正在使用兼容模式进行批量预测。如需更高效预测，请直接使用ProteinPredictorManager。")
    
    # 创建管理器并执行批量预测
    manager = ProteinPredictorManager()
    return manager.batch_predict_all(sequences, names, with_confidence, export_csv)