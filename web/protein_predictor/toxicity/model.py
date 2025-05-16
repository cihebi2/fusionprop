"""
蛋白质毒性预测模型定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFusionClassifier(nn.Module):
    """加权融合模型分类器，适合相似维度的特征"""
    def __init__(self, model_configs, hidden_dim=768, dropout=0.5):
        super().__init__()
        self.model_configs = model_configs
        self.hidden_dim = hidden_dim
        self.model_names = list(model_configs.keys())
        
        # 为每个模型创建特征归一化层
        self.layer_norms = nn.ModuleDict({
            name: nn.LayerNorm(config.output_dim)
            for name, config in model_configs.items()
        })
        
        # 为每个模型创建特征投影层
        self.projections = nn.ModuleDict({
            name: nn.Linear(config.output_dim, hidden_dim)
            for name, config in model_configs.items()
        })
        
        # 可学习的特征权重
        self.feature_weights = nn.Parameter(
            torch.ones(len(model_configs)) / len(model_configs)
        )

        # feature_encoder is removed as it's not part of the FusionModelClassifier structure from train_12_2.py
        # self.feature_encoder = nn.Sequential(...) 
        
        # 增强的预测头 - Replaced with layers matching FusionModelClassifier from train_12_2.py
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        # self.head = nn.Sequential(...) # Old head removed
        
        self.dropout_layer = nn.Dropout(dropout) # Matches self.dropout = nn.Dropout(dropout) in train_12_2.py's FusionModelClassifier
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, esm2_features=None, esmc_features=None, mask=None):
        """修改后的前向传播方法，适配预测时的调用方式"""
        # 构建模拟的batch字典
        batch = {}
        
        # 如果传入的是分开的特征，构建兼容的batch字典
        if esm2_features is not None and esmc_features is not None:
            # 计算全局表示
            if mask is not None:
                # 应用掩码计算平均值
                mask_float = mask.unsqueeze(-1).float()
                valid_tokens = mask_float.sum(dim=1).clamp(min=1)
                esm2_global = (esm2_features * mask_float).sum(dim=1) / valid_tokens
                esmc_global = (esmc_features * mask_float).sum(dim=1) / valid_tokens
            else:
                # 无掩码，直接平均
                esm2_global = esm2_features.mean(dim=1)
                esmc_global = esmc_features.mean(dim=1)
            
            # 添加到batch字典
            batch["esm2_global"] = esm2_global
            batch["esmc_global"] = esmc_global
            
            # 如果需要，也可以添加残基表示
            batch["esm2_residue"] = esm2_features
            batch["esmc_residue"] = esmc_features
            
            # 添加掩码
            if mask is not None:
                batch["esm2_mask"] = mask
                batch["esmc_mask"] = mask
        
        # 处理每个模型的输入并投影到相同维度
        projected_features = []
        
        for model_name in self.model_names:
            # 从输入获取该模型的特征
            global_key = f"{model_name}_global"
            residue_key = f"{model_name}_residue"
            mask_key = f"{model_name}_mask"
            
            if global_key in batch:
                # 使用预计算的全局表示
                global_repr = batch[global_key]
            else:
                # 从残基表示计算全局表示
                residue_repr = batch[residue_key]
                mask = batch.get(mask_key)
                
                if mask is not None:
                    mask = mask.unsqueeze(-1).float()
                    valid_tokens = mask.sum(dim=1).clamp(min=1)
                    global_repr = (residue_repr * mask).sum(dim=1) / valid_tokens
                else:
                    global_repr = residue_repr.mean(dim=1)
            
            # 特征归一化
            norm_repr = self.layer_norms[model_name](global_repr)
            
            # 投影
            proj_repr = self.projections[model_name](norm_repr)
            
            # 应用统一的编码层 - REMOVED self.feature_encoder call
            # encoded_repr = self.feature_encoder(proj_repr) 
            # projected_features.append(encoded_repr)
            projected_features.append(proj_repr) # Append proj_repr directly
        
        # 加权融合
        stacked_features = torch.stack(projected_features, dim=1)  # [B, num_models, D]
        weights = F.softmax(self.feature_weights, dim=0)  # [num_models]
        weighted_sum = (stacked_features * weights.view(1, -1, 1)).sum(dim=1)  # [B, D]
        
        # 融合特征归一化和预测
        x = self.fusion_norm(weighted_sum)
        # x = self.head(x) # Old head call removed

        # New head logic matching FusionModelClassifier from train_12_2.py
        x = self.dropout_layer(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout_layer(x) # Dropout applied again
        x = self.fc2(x)
        
        return x.squeeze(-1)  # 返回logits