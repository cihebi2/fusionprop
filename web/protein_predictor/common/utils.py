"""
蛋白质预测共享工具函数
"""
import warnings
import logging

# 通用日志配置
def setup_logger(name):
    """设置和返回日志记录器"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # 如果日志记录器还没有处理程序，则进行配置
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logger

def set_weights_only_warning():
    """设置torch.load的警告行为"""
    warnings.filterwarnings(
        "ignore", 
        message="You are using `torch.load` with `weights_only=False`",
        category=UserWarning
    )