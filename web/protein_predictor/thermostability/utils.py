"""
蛋白质热稳定性预测工具函数
"""
from ..common.utils import setup_logger

# 创建日志记录器
logger = setup_logger("ThermostabilityPredictor")

def interpret_thermostability(thermostability_score: float) -> str:
    """解释热稳定性分数
    
    Args:
        thermostability_score: 热稳定性分数(0-102)
        
    Returns:
        str: 可读的解释
    """
    # 热稳定性范围为0-102，调整阈值
    if thermostability_score >= 80:
        return "极高热稳定性 - 蛋白质很可能在高温环境下保持稳定"
    elif thermostability_score >= 60:
        return "良好热稳定性 - 蛋白质可能具有较好的热稳定性能"
    elif thermostability_score >= 40:
        return "中等热稳定性 - 蛋白质具有一般的热稳定性"
    elif thermostability_score >= 20:
        return "较低热稳定性 - 蛋白质在高温环境可能不稳定"
    else:
        return "极低热稳定性 - 蛋白质很可能在高温下迅速变性"