import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import warnings
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
import matplotlib.pyplot as plt
import datetime

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

# 改进的配置类 - 添加特征归一化选项
class NormalizedConfig:
    def __init__(self, **kwargs):
        # 特征路径
        self.esm2_train_dir = "./esm2_features_train"  
        self.esm2_test_dir = "./esm2_features_test"    
        self.esmc_train_dir = "./esmc_features_train"   
        self.esmc_test_dir = "./esmc_features_test"     
        self.model_save_dir = "./fusion_models_5_5_4_2_3"   
        
        # 添加酿酒酵母测试集路径
        self.esm2_cerevisiae_dir = "./esm2_features_cerevisiae/"  # S-PLM酿酒酵母测试特征目录
        self.esmc_cerevisiae_dir = "./esmc_features_cerevisiae/"     # ESM-C酿酒酵母测试特征目录 
        
        # 训练参数 - 默认值，可以通过kwargs修改
        self.batch_size = 32
        self.epochs = 15
        self.lr = 5e-5
        self.weight_decay = 1e-6
        self.max_seq_len = 1400
        
        # 模型参数 - 默认值，可以通过kwargs修改
        self.esm2_dim = 1280
        self.esmc_dim = 1152
        self.hidden_dim = 512
        self.dropout = 0.1
        self.head_dropout = 0.2
        
        # 特征归一化参数 - 默认值，可以通过kwargs修改
        self.normalize_features = True
        self.normalization_method = "global"
        # 预计算的统计值 (将在首次运行数据集时填充)
        self.esm2_mean = 0.0
        self.esm2_std = 1.0
        self.esmc_mean = 0.0
        self.esmc_std = 1.0
        
        # 训练设置
        self.use_amp = True
        self.grad_clip = 0.5
        self.num_workers = 6
        self.num_folds = 5
        self.random_seed = 42
        self.warmup_ratio = 0.1
        self.patience = 3    # 早停耐心值减少以加速优化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 调试选项
        self.visualize_features = False

        # 使用kwargs更新配置
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"警告: 配置中不存在属性 '{key}'")

    def set_seed(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 其他类和函数保持不变
# NormalizedDataset类、normalized_collate_fn函数、WeightedFusionModel类都保持原样
class NormalizedDataset(Dataset):
    def __init__(self, esm2_dir, esmc_dir, config, debug=True, compute_stats=False):
        self.esm2_dir = esm2_dir
        self.esmc_dir = esmc_dir
        self.config = config
        self.debug = debug
        
        # 找出共有样本ID
        self.esm2_files = {f.split('_features')[0]: f for f in os.listdir(esm2_dir) if f.endswith("_features.npy")}
        self.esmc_files = {f.split('_features')[0]: f for f in os.listdir(esmc_dir) if f.endswith("_features.npy")}
        self.common_ids = sorted(list(set(self.esm2_files.keys()) & set(self.esmc_files.keys())))
        print(f"找到 {len(self.common_ids)} 个共有样本")
        
        # 计算特征统计信息
        if compute_stats:
            self._compute_feature_stats()
        
        # 首个样本分析
        if debug:
            self._analyze_sample(0)

    def _compute_feature_stats(self):
        """计算整个数据集的特征统计信息"""
        print("计算特征统计信息...")
        
        # 收集样本
        esm2_samples = []
        esmc_samples = []
        
        # 限制样本数量以加快计算
        sample_count = min(100, len(self.common_ids))
        
        for idx in tqdm(range(sample_count), desc="收集特征样本"):
            try:
                # 加载单个样本
                sample_id = self.common_ids[idx]
                esm2_data = np.load(os.path.join(self.esm2_dir, self.esm2_files[sample_id]), allow_pickle=True).item()
                esmc_data = np.load(os.path.join(self.esmc_dir, self.esmc_files[sample_id]), allow_pickle=True).item()
                
                # 提取特征
                esm2_features = esm2_data["residue_representation"]
                esmc_features = esmc_data["residue_representation"]
                mask = esm2_data["mask"]
                
                # 仅收集有效位置的特征
                valid_esm2 = esm2_features[mask]
                valid_esmc = esmc_features[mask]
                
                esm2_samples.append(valid_esm2)
                esmc_samples.append(valid_esmc)
                
            except Exception as e:
                if self.debug:
                    print(f"处理样本 {idx} 时出错: {str(e)}")
        
        # 合并样本并计算统计信息
        all_esm2 = np.vstack(esm2_samples) if esm2_samples else np.array([])
        all_esmc = np.vstack(esmc_samples) if esmc_samples else np.array([])
        
        if len(all_esm2) > 0:
            self.config.esm2_mean = float(np.mean(all_esm2))
            self.config.esm2_std = float(np.std(all_esm2) + 1e-6)
            self.config.esmc_mean = float(np.mean(all_esmc))
            self.config.esmc_std = float(np.std(all_esmc) + 1e-6)
            
            print(f"esm2特征统计: 均值={self.config.esm2_mean:.4f}, 标准差={self.config.esm2_std:.4f}")
            print(f"ESMC特征统计: 均值={self.config.esmc_mean:.4f}, 标准差={self.config.esmc_std:.4f}")
        else:
            print("无法计算统计信息，使用默认值")

    def _analyze_sample(self, idx):
        """详细分析指定样本，特别关注特征分布"""
        if idx >= len(self.common_ids):
            print("样本索引超出范围")
            return
            
        sample_id = self.common_ids[idx]
        print(f"\n===== 样本分析: {sample_id} =====")
        
        try:
            # 加载原始特征
            esm2_data = np.load(os.path.join(self.esm2_dir, self.esm2_files[sample_id]), allow_pickle=True).item()
            esmc_data = np.load(os.path.join(self.esmc_dir, self.esmc_files[sample_id]), allow_pickle=True).item()
            
            # 输出键名
            print("esm2数据键:", list(esm2_data.keys()))
            print("ESMC数据键:", list(esmc_data.keys()))
            
            # 特征形状
            esm2_features = esm2_data["residue_representation"]
            esmc_features = esmc_data["residue_representation"]
            print(f"esm2特征形状: {esm2_features.shape}, 类型: {esm2_features.dtype}")
            print(f"ESMC特征形状: {esmc_features.shape}, 类型: {esmc_features.dtype}")
            
            # 原始特征统计
            print(f"esm2特征: 最小值={np.min(esm2_features):.4f}, 最大值={np.max(esm2_features):.4f}")
            print(f"esm2特征: 均值={np.mean(esm2_features):.4f}, 标准差={np.std(esm2_features):.4f}")
            print(f"ESMC特征: 最小值={np.min(esmc_features):.4f}, 最大值={np.max(esmc_features):.4f}")
            print(f"ESMC特征: 均值={np.mean(esmc_features):.4f}, 标准差={np.std(esmc_features):.4f}")
            
            # 掩码信息
            mask = esm2_data["mask"]
            print(f"掩码形状: {mask.shape}, 类型: {mask.dtype}")
            print(f"有效位置比例: {np.mean(mask):.4f}")
            
            # 标签信息
            sol = esm2_data["solubility"]
            print(f"溶解度标签: {sol:.4f}")
            
            # 获取归一化版本的特征
            normalized_esm2 = self._normalize_features(torch.from_numpy(esm2_features).float(), "esm2")
            normalized_esmc = self._normalize_features(torch.from_numpy(esmc_features).float(), "esmc")
            
            # 打印归一化后的统计信息
            print(f"\n归一化后esm2特征: 均值={normalized_esm2.mean().item():.4f}, 标准差={normalized_esm2.std().item():.4f}")
            print(f"归一化后ESMC特征: 均值={normalized_esmc.mean().item():.4f}, 标准差={normalized_esmc.std().item():.4f}")
            
            # 可视化特征分布
            if self.config.visualize_features:
                self._visualize_feature_distribution(
                    esm2_features, esmc_features, 
                    normalized_esm2.numpy(), normalized_esmc.numpy(),
                    sample_id
                )
            
        except Exception as e:
            print(f"分析出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("=============================")

    def _visualize_feature_distribution(self, raw_esm2, raw_esmc, norm_esm2, norm_esmc, sample_id):
        """可视化特征分布"""
        try:
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 原始特征分布
            axes[0,0].hist(raw_esm2.flatten(), bins=50, alpha=0.7)
            axes[0,0].set_title('原始esm2特征分布')
            axes[0,0].grid(True)
            
            axes[0,1].hist(raw_esmc.flatten(), bins=50, alpha=0.7)
            axes[0,1].set_title('原始ESMC特征分布')
            axes[0,1].grid(True)
            
            # 归一化后特征分布
            axes[1,0].hist(norm_esm2.flatten(), bins=50, alpha=0.7)
            axes[1,0].set_title('归一化后esm2特征分布')
            axes[1,0].grid(True)
            
            axes[1,1].hist(norm_esmc.flatten(), bins=50, alpha=0.7)
            axes[1,1].set_title('归一化后ESMC特征分布')
            axes[1,1].grid(True)
            
            # 添加统计信息
            plt.suptitle(f"样本 {sample_id} 特征分布对比\n"
                        f"原始esm2: μ={np.mean(raw_esm2):.3f}, σ={np.std(raw_esm2):.3f} | "
                        f"原始ESMC: μ={np.mean(raw_esmc):.3f}, σ={np.std(raw_esmc):.3f}\n"
                        f"归一化esm2: μ={np.mean(norm_esm2):.3f}, σ={np.std(norm_esm2):.3f} | "
                        f"归一化ESMC: μ={np.mean(norm_esmc):.3f}, σ={np.std(norm_esmc):.3f}",
                        fontsize=12)
            
            plt.tight_layout()
            
            # 创建图表保存目录
            vis_dir = Path("feature_visualizations")
            vis_dir.mkdir(exist_ok=True)
            
            # 保存图表
            plt.savefig(vis_dir / f"sample_{sample_id}_feature_distribution.png", dpi=100)
            plt.close()
            
            print(f"特征分布图表已保存到 feature_visualizations/sample_{sample_id}_feature_distribution.png")
            
        except Exception as e:
            print(f"可视化特征分布失败: {str(e)}")

    def _normalize_features(self, features, feature_type="esm2"):
        """根据配置的方法归一化特征"""
        # 如果禁用归一化，直接返回原始特征
        if not self.config.normalize_features:
            return features
            
        # 基于方法选择归一化策略
        if self.config.normalization_method == "global":
            # 使用全局均值和标准差
            if feature_type == "esm2":
                return (features - self.config.esm2_mean) / self.config.esm2_std
            else:  # esmc
                return (features - self.config.esmc_mean) / self.config.esmc_std
                
        elif self.config.normalization_method == "sequence":
            # 对每个序列单独归一化
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True) + 1e-6
            return (features - mean) / std
            
        # 默认情况：不归一化
        return features

    def __len__(self):
        return len(self.common_ids)
    
    def __getitem__(self, idx):
        sample_id = self.common_ids[idx]
        
        try:
            # 加载特征
            esm2_data = np.load(os.path.join(self.esm2_dir, self.esm2_files[sample_id]), allow_pickle=True).item()
            esmc_data = np.load(os.path.join(self.esmc_dir, self.esmc_files[sample_id]), allow_pickle=True).item()
            
            # 获取特征和掩码
            esm2_features = torch.from_numpy(esm2_data["residue_representation"]).float()
            esmc_features = torch.from_numpy(esmc_data["residue_representation"]).float()
            
            # 维度规范化
            if esm2_features.dim() > 2:
                esm2_features = esm2_features.squeeze(0)
            if esmc_features.dim() > 2:
                esmc_features = esmc_features.squeeze(0)
                
            # 确保维度正确
            assert esm2_features.dim() == 2, f"esm2特征维度错误: {esm2_features.shape}"
            assert esmc_features.dim() == 2, f"ESMC特征维度错误: {esmc_features.shape}"
            
            # 特征归一化
            esm2_features = self._normalize_features(esm2_features, "esm2")
            esmc_features = self._normalize_features(esmc_features, "esmc")
                
            # 使用S-PLM的掩码
            mask = torch.from_numpy(esm2_data["mask"]).bool()
            if mask.dim() > 1:
                mask = mask.squeeze(0)
                
            solubility = torch.tensor(esm2_data["solubility"]).float().clamp(0.0, 1.0)
            
            return esm2_features, esmc_features, mask, solubility
            
        except Exception as e:
            if self.debug:
                print(f"加载样本 {sample_id} 出错: {str(e)}")
            # 返回一个小尺寸的dummy样本
            return torch.zeros(10, self.config.esm2_dim), torch.zeros(10, self.config.esmc_dim), torch.zeros(10, dtype=torch.bool), torch.tensor(0.5)

def normalized_collate_fn(batch):
    """加强健壮性的批处理函数"""
    esm2_features, esmc_features, masks, solubilities = zip(*batch)
    
    # 查找当前批次中的最大序列长度
    max_len = max(feat.size(0) for feat in esm2_features)
    
    # 填充批次中的每个序列到相同长度
    padded_esm2 = []
    padded_esmc = []
    padded_masks = []
    
    for i in range(len(esm2_features)):
        curr_len = esm2_features[i].size(0)
        
        # 创建填充张量
        esm2_pad = torch.zeros(max_len, esm2_features[i].size(1))
        esmc_pad = torch.zeros(max_len, esmc_features[i].size(1))
        mask_pad = torch.zeros(max_len, dtype=torch.bool)
        
        # 复制数据到填充张量
        esm2_pad[:curr_len] = esm2_features[i]
        esmc_pad[:curr_len] = esmc_features[i]
        mask_pad[:curr_len] = masks[i]
        
        padded_esm2.append(esm2_pad)
        padded_esmc.append(esmc_pad)
        padded_masks.append(mask_pad)
    
    return {
        "esm2_features": torch.stack(padded_esm2),
        "esmc_features": torch.stack(padded_esmc),
        "mask": torch.stack(padded_masks),
        "solubility": torch.stack(solubilities)
    }

class WeightedFusionModel(nn.Module):
    def __init__(self, esm2_dim=1280, esmc_dim=1152, hidden_dim=512, dropout=0.1, use_layer_norm=True):
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
    
    def forward(self, esm2_features, esmc_features, mask):
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
# 修改后的训练器类 - 只针对加权融合模型且不保存模型
class OptimizedTrainer:
    def __init__(self, config):
        """
        初始化优化训练器
        Args:
            config: 配置对象
        """
        self.config = config
        self.device = config.device
        self.model_type = 'weighted'  # 只使用加权融合
        self.scaler = GradScaler(enabled=config.use_amp)
        
        # 创建结果目录
        self.results_dir = Path(config.model_save_dir) / "results"  # 修改结果目录
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化结果字典
        self._initialize_results()

    
    def _initialize_results(self):
        """初始化结果字典"""
        self.results = {
            "model_type": "weighted_fusion",
            "hyperparameters": {
                "hidden_dim": self.config.hidden_dim,
                "dropout": self.config.dropout,
                "lr": self.config.lr,
                "batch_size": self.config.batch_size,
                "weight_decay": self.config.weight_decay,
                "normalization_method": self.config.normalization_method
            },
            "folds": [],
            "fold_test_r2": [],
            "fold_val_r2": [],
            "fold_cerevisiae_r2": []
        }
    
    def train_kfold(self):
        """执行K折交叉验证"""
        config = self.config
        
        # 创建完整数据集
        full_dataset = NormalizedDataset(
            esm2_dir=config.esm2_train_dir,
            esmc_dir=config.esmc_train_dir,
            config=config,
            debug=False,
            compute_stats=True  # 计算全局特征统计信息
        )
        
        # 创建K折分割器
        kf = KFold(n_splits=config.num_folds, shuffle=True, random_state=config.random_seed)
        
        # 记录所有折的结果
        all_val_r2 = []
        all_test_r2 = []
        all_cerevisiae_r2 = []
        
        # 训练每个折
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_dataset)))):
            fold_num = fold + 1
            print(f"\n🔢 训练Fold {fold_num}/{config.num_folds}")
            
            # 创建数据子集
            train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
            val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=normalized_collate_fn,
                num_workers=config.num_workers,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size * 2,
                shuffle=False,
                collate_fn=normalized_collate_fn,
                num_workers=config.num_workers,
                pin_memory=True
            )
            
            # 创建模型
            model = WeightedFusionModel(
                esm2_dim=config.esm2_dim,
                esmc_dim=config.esmc_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
                use_layer_norm=True
            ).to(self.device)
            
            # 训练模型
            val_metrics = self._train_fold(model, train_loader, val_loader, fold_num)
            
            # 在测试集上评估
            test_metrics = self._evaluate_test_set(model)
            
            # 在酿酒酵母测试集上评估
            cerevisiae_metrics = self._evaluate_cerevisiae_set(model)
            
            # 记录结果
            fold_entry = {
                "fold_number": fold_num,
                "validation_r2": val_metrics["r2"],
                "validation_rmse": val_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "test_rmse": test_metrics["rmse"],
                "cerevisiae_r2": cerevisiae_metrics["r2"],
                "cerevisiae_rmse": cerevisiae_metrics["rmse"],
                "feature_weights": {
                    "esm2_weight": float(torch.sigmoid(model.alpha).item()),
                    "esmc_weight": float(1 - torch.sigmoid(model.alpha).item())
                }
            }
            
            self.results["folds"].append(fold_entry)
            self.results["fold_test_r2"].append(test_metrics["r2"])
            self.results["fold_val_r2"].append(val_metrics["r2"])
            self.results["fold_cerevisiae_r2"].append(cerevisiae_metrics["r2"])
            
            all_val_r2.append(val_metrics["r2"])
            all_test_r2.append(test_metrics["r2"])
            all_cerevisiae_r2.append(cerevisiae_metrics["r2"])
            
            # 输出当前fold结果
            alpha = torch.sigmoid(model.alpha).item()
            print(f"✅ Fold {fold_num} 结果：")
            print(f"   验证 R²: {val_metrics['r2']:.4f}")
            print(f"   测试 R²: {test_metrics['r2']:.4f}") 
            print(f"   酿酒酵母 R²: {cerevisiae_metrics['r2']:.4f}")
            print(f"   特征权重: esm2 = {alpha:.4f}, ESM-C = {1-alpha:.4f}")
        
        # 计算平均性能
        avg_val_r2 = np.mean(all_val_r2)
        avg_test_r2 = np.mean(all_test_r2)
        std_val_r2 = np.std(all_val_r2)
        std_test_r2 = np.std(all_test_r2)
        avg_cerevisiae_r2 = np.mean(all_cerevisiae_r2)
        std_cerevisiae_r2 = np.std(all_cerevisiae_r2)

        # 保存聚合结果
        self.results["average_validation_r2"] = float(avg_val_r2)
        self.results["std_validation_r2"] = float(std_val_r2)
        self.results["average_test_r2"] = float(avg_test_r2)
        self.results["std_test_r2"] = float(std_test_r2)
        self.results["average_cerevisiae_r2"] = float(avg_cerevisiae_r2)
        self.results["std_cerevisiae_r2"] = float(std_cerevisiae_r2)

        # 打印总体结果
        print(f"\n📊 加权融合模型整体结果:")
        print(f"   验证集 R²: {avg_val_r2:.4f} ± {std_val_r2:.4f}")
        print(f"   测试集 R²: {avg_test_r2:.4f} ± {std_test_r2:.4f}")
        print(f"   酿酒酵母测试集 R²: {avg_cerevisiae_r2:.4f} ± {std_cerevisiae_r2:.4f}")
        
        # 保存结果到文件
        self._save_results()
        
        return self.results
    
    def _train_fold(self, model, train_loader, val_loader, fold_num):
        """训练单个折的模型"""
        config = self.config
        device = self.device
        
        # 创建优化器
        optimizer = AdamW(
            model.parameters(), 
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # 创建学习率调度器
        total_steps = len(train_loader) * config.epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.lr,
            total_steps=total_steps,
            pct_start=config.warmup_ratio,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        # 跟踪最佳模型
        best_val_r2 = -float('inf')
        best_model_state = None
        patience_counter = 0
        
        # 训练循环
        for epoch in range(1, config.epochs + 1):
            # 训练阶段
            model.train()
            epoch_losses = []
            
            for batch in train_loader:
                # 准备数据
                esm2_features = batch["esm2_features"].to(device)
                esmc_features = batch["esmc_features"].to(device)
                mask = batch["mask"].to(device)
                targets = batch["solubility"].to(device)
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播(使用混合精度)
                with autocast(enabled=config.use_amp):
                    outputs = model(esm2_features, esmc_features, mask)
                    loss = F.mse_loss(outputs, targets)
                
                # 反向传播(使用混合精度)
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if config.grad_clip > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                
                # 更新权重
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                
                # 记录损失
                epoch_losses.append(loss.item())
            
            # 计算平均训练损失
            avg_train_loss = np.mean(epoch_losses)
            
            # 验证阶段
            val_metrics = self._evaluate_model(model, val_loader)
            val_r2 = val_metrics["r2"]
            val_rmse = val_metrics["rmse"]
            
            # 输出当前训练状态
            print(f"Epoch {epoch} | Loss: {avg_train_loss:.4f} | Val R²: {val_r2:.4f} | Val RMSE: {val_rmse:.4f}")
            
            # 检查是否是最佳模型
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f"💾 新的最佳模型! R²: {val_r2:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"⏹️ 早停: {patience_counter}个epoch无改善")
                    break
        
        # 加载最佳模型状态
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 最终验证集评估
        final_val_metrics = self._evaluate_model(model, val_loader)

        # 保存最佳模型
        model_save_path = self.results_dir / f"best_model_fold_{fold_num}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"💾 第 {fold_num} 折的最佳模型已保存至: {model_save_path}")
        
        return final_val_metrics
    
    def _evaluate_model(self, model, loader):
        """评估模型性能"""
        model.eval()
        device = self.device
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in loader:
                # 准备数据
                esm2_features = batch["esm2_features"].to(device)
                esmc_features = batch["esmc_features"].to(device)
                mask = batch["mask"].to(device)
                targets = batch["solubility"].cpu().numpy()
                
                # 前向传播
                with autocast(enabled=self.config.use_amp):
                    outputs = model(esm2_features, esmc_features, mask).cpu().numpy()
                
                # 收集预测和目标
                all_preds.append(outputs)
                all_targets.append(targets)
        
        # 合并所有预测和目标
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # 过滤无效值
        valid_mask = ~np.isnan(all_preds) & ~np.isnan(all_targets)
        clean_preds = all_preds[valid_mask]
        clean_targets = all_targets[valid_mask]
        
        # 计算指标
        try:
            r2 = r2_score(clean_targets, clean_preds)
            r2 = max(min(r2, 1.0), -1.0)  # 约束R²范围
        except:
            r2 = 0.0
        
        rmse = np.sqrt(mean_squared_error(clean_targets, clean_preds))
        
        # 如果是加权融合模型，报告权重
        if hasattr(model, 'alpha'):
            alpha = torch.sigmoid(model.alpha).item()
            print(f"🔄 特征权重: esm2 = {alpha:.4f}, ESM-C = {1-alpha:.4f}")
        
        return {"r2": float(r2), "rmse": float(rmse)}
    
    def _evaluate_test_set(self, model):
        """在测试集上评估模型"""
        test_dataset = NormalizedDataset(
            esm2_dir=self.config.esm2_test_dir,
            esmc_dir=self.config.esmc_test_dir,
            config=self.config,
            debug=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            collate_fn=normalized_collate_fn,
            num_workers=self.config.num_workers
        )
        
        return self._evaluate_model(model, test_loader)
        
    def _evaluate_cerevisiae_set(self, model):
        """在酿酒酵母测试集上评估模型"""
        cerevisiae_dataset = NormalizedDataset(
            esm2_dir=self.config.esm2_cerevisiae_dir,
            esmc_dir=self.config.esmc_cerevisiae_dir,
            config=self.config,
            debug=False
        )
        
        cerevisiae_loader = DataLoader(
            cerevisiae_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            collate_fn=normalized_collate_fn,
            num_workers=self.config.num_workers
        )
        
        return self._evaluate_model(model, cerevisiae_loader)
    
    def _save_results(self):
        """将结果保存到JSON文件"""
        results_file = self.results_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=4)
        print(f"结果已保存到 {results_file}")

# 移除超参数优化函数
# def optimize_hyperparameters(n_trials=30):
#     pass

if __name__ == "__main__":
    # 设置matplotlib中文支持
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    
    # 指定模型参数
    model_params = {
        "hidden_dim": 256,
        "dropout": 0.5,
        "lr": 5e-5,
        "batch_size": 16,
        "weight_decay": 2e-5,
        "normalization_method": "sequence"
    }
    
    # 创建配置实例
    config = NormalizedConfig(**model_params)
    config.set_seed()
    
    # 创建训练器
    trainer = OptimizedTrainer(config)
    
    # 训练模型
    detailed_results = trainer.train_kfold()
    
    # 打印最终结果摘要
    print("\n📊 最终模型结果摘要:")
    print(f"验证集 R²: {detailed_results['average_validation_r2']:.4f}")
    print(f"测试集 R²: {detailed_results['average_test_r2']:.4f} ± {detailed_results['std_test_r2']:.4f}")
    print(f"酿酒酵母测试集 R²: {detailed_results['average_cerevisiae_r2']:.4f} ± {detailed_results['std_cerevisiae_r2']:.4f}")