"""
蛋白质毒性预测工具函数
"""
from typing import Dict
from ..common.utils import setup_logger

# 创建日志记录器
logger = setup_logger("ToxicityPredictor")

def interpret_toxicity(toxicity_prob: float) -> Dict:
    """解释毒性预测结果
    
    Args:
        toxicity_prob: 毒性概率 (0-1)
        
    Returns:
        Dict: 结果解释字典
    """
    # 定义不同概率区间的解释
    if toxicity_prob >= 0.9:
        risk_level = "极高"
        description = "高度可能是有毒蛋白质，具有很高的生物毒性风险"
    elif toxicity_prob >= 0.7:
        risk_level = "高"
        description = "很可能是有毒蛋白质，有较高的生物毒性风险"
    elif toxicity_prob >= 0.5:
        risk_level = "中等"
        description = "可能是有毒蛋白质，具有一定的生物毒性风险"
    elif toxicity_prob >= 0.3:
        risk_level = "低"
        description = "可能是无毒蛋白质，但仍有少量生物毒性风险"
    else:
        risk_level = "极低"
        description = "高度可能是无毒蛋白质，生物毒性风险极低"
    
    # 分类结果（二分类）
    is_toxic = toxicity_prob >= 0.5
    classification = "有毒蛋白质" if is_toxic else "无毒蛋白质"
    
    return {
        "risk_level": risk_level,
        "description": description,
        "classification": classification,
        "is_toxic": is_toxic
    }