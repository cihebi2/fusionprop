"""
蛋白质毒性预测库 - 使用深度学习预测蛋白质序列的毒性

示例用法:
    from protein_toxicity_predictor import predict_toxicity
    
    # 单序列预测
    result = predict_toxicity("蛋白质序列", return_confidence=True)
    print(f"毒性概率: {result['toxicity_probability']}")
    print(f"风险级别: {result['risk_level']}")
    
    # 批量预测
    from protein_toxicity_predictor import batch_predict
    sequences = ["序列1", "序列2", "序列3"]
    names = ["蛋白质1", "蛋白质2", "蛋白质3"]
    results = batch_predict(sequences, names, export_csv="results.csv")
"""

# 导入主要函数，使其可以直接从包中访问
from .predictor import predict_toxicity, batch_predict

# 导出版本信息
__version__ = "1.0.0"