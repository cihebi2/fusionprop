"""
蛋白质溶解性预测工具函数
"""
import logging
import warnings

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SolubilityPredictor")

def interpret_solubility(solubility_score: float) -> str:
    """解释溶解度分数
    
    Args:
        solubility_score: 溶解度分数(0-1)
        
    Returns:
        str: 可读的解释
    """
    # 溶解度范围为0-1，调整阈值分类
    if solubility_score >= 0.8:
        return "极高溶解度 - 蛋白质在水溶液中有极高的溶解性，有利于表达和纯化"
    elif solubility_score >= 0.6:
        return "高溶解度 - 蛋白质在水溶液中具有良好的溶解性，较易表达和纯化"
    elif solubility_score >= 0.4:
        return "中等溶解度 - 蛋白质具有一般的溶解性能，表达和纯化可能需要一定优化"
    elif solubility_score >= 0.2:
        return "低溶解度 - 蛋白质在水溶液中溶解性较差，可能需要特殊条件或添加剂辅助溶解"
    else:
        return "极低溶解度 - 蛋白质很可能形成包含体或难以溶解，表达和纯化挑战较大"