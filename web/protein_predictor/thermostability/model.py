"""
蛋白质热稳定性预测模型定义
"""
import torch
import torch.nn as nn

class WeightedFusionRegressor(nn.Module):
    """加权融合模型回归器，用于热稳定性预测"""
    def __init__(self, model_configs, hidden_dim=512, dropout=0.1, use_layer_norm=True):
        super().__init__()
        self.model_names = [name for name, config in model_configs.items() if config.enabled]
        self.output_dims = {name: config.output_dim for name, config in model_configs.items() if config.enabled}
        
        # 确保至少有一个模型被启用
        assert len(self.model_names) > 0, "至少需要一个启用的模型"
        
        # 确保只有两个模型被使用（ESM2和ESMC）
        if len(self.model_names) > 2:
            raise ValueError("当前实现仅支持两个模型的融合")
        
        # 记录模型名称
        self.esm2_name = self.model_names[0]  # 假设第一个是 ESM2
        self.esmc_name = self.model_names[1] if len(self.model_names) > 1 else None  # 第二个是 ESMC
        
        # 获取输入维度
        self.esm2_dim = self.output_dims[self.esm2_name]
        self.esmc_dim = self.output_dims[self.esmc_name] if self.esmc_name else 0
        
        self.use_layer_norm = use_layer_norm
        
        # 可选的层归一化
        if use_layer_norm:
            self.esm2_norm = nn.LayerNorm(self.esm2_dim)
            if self.esmc_name:
                self.esmc_norm = nn.LayerNorm(self.esmc_dim)
        
        # 特征投影层
        self.esm2_proj = nn.Linear(self.esm2_dim, hidden_dim)
        if self.esmc_name:
            self.esmc_proj = nn.Linear(self.esmc_dim, hidden_dim)
        
        # 特征编码层 - 为每个模型分别添加
        self.esm2_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        if self.esmc_name:
            self.esmc_encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
        
        # 可学习特征权重参数
        self.alpha = nn.Parameter(torch.tensor([0.5]))
        
        # 预测头
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
        )
        
        # 保存归一化后的权重
        self.normalized_weights = None
        
    def forward(self, esm2_features, esmc_features, mask=None):
        """前向传播"""
        # 应用层归一化（如果启用）
        if self.use_layer_norm:
            esm2_features = self.esm2_norm(esm2_features)
            if self.esmc_name:
                esmc_features = self.esmc_norm(esmc_features)
        
        # 特征投影
        esm2_proj = self.esm2_proj(esm2_features)
        esm2_encoded = self.esm2_encoder(esm2_proj)
        
        if self.esmc_name:
            esmc_proj = self.esmc_proj(esmc_features)
            esmc_encoded = self.esmc_encoder(esmc_proj)
            
            # 计算加权平均特征
            alpha = torch.sigmoid(self.alpha)  # 确保权重在0-1之间
            self.normalized_weights = alpha.item()  # 存储当前权重
            
            # 加权融合
            fused_features = alpha * esm2_encoded + (1 - alpha) * esmc_encoded
        else:
            # 如果只有ESM2，直接使用其编码
            fused_features = esm2_encoded
            self.normalized_weights = 1.0
        
        # 处理掩码（如果提供）
        if mask is not None:
            # 将掩码扩展为与特征相同的维度
            extended_mask = mask.unsqueeze(-1).expand_as(fused_features)
            
            # 应用掩码并计算平均特征
            masked_features = fused_features * extended_mask
            feature_sum = masked_features.sum(dim=1)
            mask_sum = extended_mask.sum(dim=1)
            # 防止除零
            mask_sum = torch.clamp(mask_sum, min=1.0)
            sequence_features = feature_sum / mask_sum
        else:
            # 无掩码时简单平均
            sequence_features = fused_features.mean(dim=1)
        
        # 通过预测头
        output = self.head(sequence_features)
        return output.squeeze(-1)