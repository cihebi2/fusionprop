"""
蛋白质溶解性预测模型定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFusionModel(nn.Module):
    """加权融合模型，用于蛋白质溶解性预测"""
    def __init__(self, esm2_dim=1280, esmc_dim=1152, hidden_dim=256, dropout=0.5, use_layer_norm=True):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        
        # 可选的层归一化
        if use_layer_norm:
            self.esm2_norm = nn.LayerNorm(esm2_dim)
            self.esmc_norm = nn.LayerNorm(esmc_dim)
        
        # 特征投影层
        self.esm2_proj = nn.Linear(esm2_dim, hidden_dim)
        self.esmc_proj = nn.Linear(esmc_dim, hidden_dim)
        
        # 可学习特征权重
        self.alpha = nn.Parameter(torch.tensor([0.5]))
        
        # 预测头
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, esm2_features, esmc_features, mask=None):
        """前向传播"""
        # 可选的层归一化
        if self.use_layer_norm:
            esm2_features = self.esm2_norm(esm2_features)
            esmc_features = self.esmc_norm(esmc_features)
        
        # 投影到相同维度
        p_esm2 = self.esm2_proj(esm2_features)
        p_esmc = self.esmc_proj(esmc_features)
        
        # 池化
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            valid_tokens = mask.sum(dim=1).clamp(min=1)
            pooled_esm2 = (p_esm2 * mask).sum(dim=1) / valid_tokens
            pooled_esmc = (p_esmc * mask).sum(dim=1) / valid_tokens
        else:
            pooled_esm2 = p_esm2.mean(dim=1)
            pooled_esmc = p_esmc.mean(dim=1)
        
        # 加权融合
        alpha = torch.sigmoid(self.alpha)  # 转换到0-1范围
        weighted = alpha * pooled_esm2 + (1 - alpha) * pooled_esmc
        
        # 记录当前权重值 (仅用于分析)
        if not self.training and hasattr(self, '_current_alpha'):
            self._current_alpha = alpha.item()
        
        # 预测
        return self.head(weighted).squeeze(-1)