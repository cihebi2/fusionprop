"""
蛋白质热稳定性预测核心功能
"""
import os
import json
import torch
import numpy as np
import logging
import time
import pandas as pd
import shutil
from pathlib import Path
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from typing import Dict, List, Optional

from .config import PredictorConfig, ModelConfig
from .model import WeightedFusionRegressor
from ..common.data import FeatureDataset, feature_collate_fn, extract_features
from .utils import interpret_thermostability, logger
from ..common.utils import set_weights_only_warning
class ThermostabilityPredictor:
    """蛋白质热稳定性预测器，使用集成模型进行预测"""
    
    def __init__(self, config=None, model_dir=None):
        """初始化预测器"""
        # 初始化配置
        self.config = config if config is not None else PredictorConfig()
        if model_dir:
            self.config.model_dir = model_dir
            
        # 设置设备和随机种子
        self.device = self.config.device
        self.config.set_seed()
        
        # 初始化模型列表和路径
        self.models = []
        self.model_paths = []
        
        # 加载配置
        self._load_model_config()
        
        # 加载特征统计数据
        self._load_feature_stats()
        
        # 加载模型
        self._load_models()
        
        logger.info(f"热稳定性预测器已初始化，设备: {self.device}, 已加载 {len(self.models)} 个模型")
        
    def _load_model_config(self):
        """加载模型配置"""
        config_path = Path(self.config.model_dir) / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                    
                # 获取模型配置
                model_configs = config_data.get("model_configs", {})
                self.model_configs = {}
                
                for model_name, model_config in model_configs.items():
                    self.model_configs[model_name] = ModelConfig(
                        output_dim=model_config.get("output_dim", 1280),
                        enabled=model_config.get("enabled", True),
                        model_name=model_name
                    )
                    
                logger.info(f"成功加载模型配置: {list(self.model_configs.keys())}")
            except Exception as e:
                logger.warning(f"读取模型配置失败: {e}，使用默认配置")
                self.model_configs = {
                    "esm2": ModelConfig(output_dim=1280),
                    "esmc": ModelConfig(output_dim=1152)
                }
        else:
            logger.warning(f"模型配置文件不存在: {config_path}，使用默认配置")
            # 使用默认配置
            self.model_configs = {
                "esm2": ModelConfig(output_dim=1280),
                "esmc": ModelConfig(output_dim=1152)
            }
    
    def _load_feature_stats(self):
        """加载特征统计数据（均值、标准差）"""
        stats_path = Path(self.config.model_dir) / "feature_stats.json"
        
        if stats_path.exists():
            try:
                with open(stats_path, "r") as f:
                    stats = json.load(f)
                
                self.config.esm2_mean = stats.get("esm2_mean", 0.0)
                self.config.esm2_std = stats.get("esm2_std", 1.0)
                self.config.esmc_mean = stats.get("esmc_mean", 0.0)
                self.config.esmc_std = stats.get("esmc_std", 1.0)
                
                logger.info(f"已加载特征统计数据: ESM2 μ={self.config.esm2_mean:.4f}, σ={self.config.esm2_std:.4f}, "
                          f"ESMC μ={self.config.esmc_mean:.4f}, σ={self.config.esmc_std:.4f}")
            except Exception as e:
                logger.warning(f"无法加载特征统计数据: {e}")
        else:
            logger.warning(f"特征统计文件不存在: {stats_path}，将使用默认值")
    
    def _load_models(self):
        """加载所有保存的模型"""
        # 忽略torch.load的权重警告
        set_weights_only_warning()
        
        model_dir = Path(self.config.model_dir)
        logger.info(f"从 {model_dir} 加载训练好的模型...")
        
        # 查找所有模型文件
        model_files = list(model_dir.glob("best_model.pth"))
        
        if not model_files:
            error_msg = f"在 {model_dir} 中未找到模型文件"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"找到 {len(model_files)} 个模型文件")
        self.model_paths = model_files
        
        # 尝试从配置文件中读取hidden_dim
        config_path = Path(self.config.model_dir) / "config.json"
        hidden_dim = 256  # 默认使用训练时的hidden_dim=256
        dropout = 0.3     # 默认使用训练时的dropout=0.3
        
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                    # 从训练配置中读取hidden_dim和dropout
                    train_config = config_data.get("training_config", {})
                    hidden_dim = train_config.get("hidden_dim", 256)
                    dropout = train_config.get("dropout", 0.3)
                    logger.info(f"从配置文件读取模型参数: hidden_dim={hidden_dim}, dropout={dropout}")
            except Exception as e:
                logger.warning(f"读取配置文件参数失败: {e}，使用默认值 hidden_dim={hidden_dim}, dropout={dropout}")
        
        # 加载模型
        for path in self.model_paths:
            try:
                # 使用和训练时相同的hidden_dim和dropout创建模型实例
                model = WeightedFusionRegressor(self.model_configs, hidden_dim=hidden_dim, dropout=dropout)
                
                # 加载模型权重并提取内部的state_dict
                loaded_obj = torch.load(path, map_location=self.device)
                
                # 检查加载的对象结构并提取state_dict
                if isinstance(loaded_obj, dict) and "state_dict" in loaded_obj:
                    # 如果加载的对象是字典且包含"state_dict"键
                    state_dict = loaded_obj["state_dict"]
                    logger.info(f"从包含嵌套状态字典的模型文件中提取权重")
                else:
                    # 否则假设加载的对象直接是state_dict
                    state_dict = loaded_obj
                
                # 加载权重到模型
                model.load_state_dict(state_dict)
                
                # 将模型移动到设备上并设置为评估模式
                model = model.to(self.device)
                model.eval()
                
                self.models.append(model)
                logger.info(f"成功加载模型: {path.name}")
            except Exception as e:
                logger.error(f"加载模型 {path} 失败: {str(e)}")
        
        logger.info(f"成功加载 {len(self.models)} 个模型")
        
        # 验证是否有加载成功的模型
        if not self.models:
            raise ValueError("没有成功加载任何模型。请检查模型文件格式是否兼容。")
    
    def predict_batch(self, 
                    esm2_features_dir: str, 
                    esmc_features_dir: str, 
                    sample_ids: Optional[List[str]] = None,
                    return_confidence: bool = False) -> Dict:
        """批量预测样本热稳定性
        
        Args:
            esm2_features_dir: ESM2特征目录
            esmc_features_dir: ESMC特征目录
            sample_ids: 指定要预测的样本ID列表，None则预测所有样本
            return_confidence: 是否返回预测置信度（集成模型的标准差）
            
        Returns:
            Dict: 预测结果字典
        """
        if len(self.models) == 0:
            logger.error("[ThermostabilityPredictor.predict_batch] No models loaded, cannot predict.")
            return {"prediction_map": {}, "mean_thermostability": None}
            
        logger.info(f"[ThermostabilityPredictor.predict_batch] Called with esm2_dir: '{esm2_features_dir}', esmc_dir: '{esmc_features_dir}', sample_ids: {sample_ids}")
        try:
            # 准备数据集
            logger.info(f"[{[sid for sid in sample_ids] if sample_ids else 'All samples'}] Creating FeatureDataset with esm2_dir='{esm2_features_dir}', esmc_dir='{esmc_features_dir}'...")
            predict_dataset = FeatureDataset(
                esm2_dir=esm2_features_dir,
                esmc_dir=esmc_features_dir,
                config=self.config,
                sample_ids=sample_ids
            )
            logger.info(f"[{[sid for sid in sample_ids] if sample_ids else 'All samples'}] FeatureDataset created. Length: {len(predict_dataset)}")

            if len(predict_dataset) == 0:
                logger.warning(f"[{[sid for sid in sample_ids] if sample_ids else 'All samples'}] FeatureDataset is empty for ThermostabilityPredictor.")
                return {"prediction_map": {}, "mean_thermostability": None}

            # 准备数据加载器
            predict_loader = DataLoader(
                predict_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=feature_collate_fn,
                num_workers=0,
                persistent_workers=False
            )
            
            # 预测结果
            prediction_map = {}
            all_preds = []
            
            logger.info(f"开始预测 {len(predict_dataset)} 个样本...")
            
            # 预测
            with torch.no_grad():
                for batch in predict_loader:
                    # 移动数据到设备
                    esm2_features = batch["esm2_features"].to(self.device)
                    esmc_features = batch["esmc_features"].to(self.device)
                    masks = batch["mask"].to(self.device)
                    sample_ids = batch["sample_id"]
                    
                    # 预测
                    batch_preds = []
                    
                    # 使用每个模型进行预测
                    with autocast(enabled=self.config.use_amp):
                        for model in self.models:
                            pred = model(esm2_features, esmc_features, masks)
                            batch_preds.append(pred.cpu())
                    
                    # 计算每个样本的平均预测值和置信度（如果需要）
                    for i, sample_id in enumerate(sample_ids):
                        # 获取所有模型对当前样本的预测
                        sample_preds = [preds[i].item() for preds in batch_preds]
                        avg_pred = sum(sample_preds) / len(sample_preds)
                        
                        result = {"thermostability": avg_pred}
                        
                        # 如果需要置信度，计算模型预测的标准差
                        if return_confidence and len(sample_preds) > 1:
                            std_dev = float(np.std(sample_preds))
                            result["confidence"] = std_dev
                            
                        # 存储预测结果
                        prediction_map[sample_id] = result
                        all_preds.append(avg_pred)
            
            logger.info(f"[{[sid for sid in sample_ids] if sample_ids else 'All samples'}] Final prediction map for Thermostability generated with {len(prediction_map)} entries. Overall mean: {sum(all_preds) / len(all_preds) if all_preds else None}")

            return {
                "prediction_map": prediction_map,
                "mean_thermostability": sum(all_preds) / len(all_preds) if all_preds else None
            }
        except FileNotFoundError as fnf_error:
            logger.error(f"[ThermostabilityPredictor.predict_batch] FileNotFoundError: {fnf_error}.")
            return {"prediction_map": {}, "mean_thermostability": None, "error": str(fnf_error)}
        except Exception as e:
            logger.error(f"[ThermostabilityPredictor.predict_batch] Error during prediction: {e}", exc_info=True)
            return {"prediction_map": {}, "mean_thermostability": None, "error": str(e)}

def predict_thermostability(
    sequence: str, 
    return_confidence: bool = False,
    model_dir: Optional[str] = None,
    cleanup_features: bool = True
) -> Dict:
    """从蛋白质序列直接预测热稳定性
    
    Args:
        sequence: 蛋白质序列字符串
        return_confidence: 是否返回预测置信度
        model_dir: 模型目录，None则使用默认
        cleanup_features: 完成后是否删除临时特征文件
        
    Returns:
        Dict: 包含热稳定性预测结果的字典
    """
    try:
        # 1. 提取特征
        temp_dir = Path("./temp_features")
        feature_result = extract_features(sequence, output_dir=str(temp_dir))
        
        # 验证特征提取结果
        if not isinstance(feature_result, dict):
            raise TypeError(f"特征提取结果应为字典，实际为 {type(feature_result)}")
            
        if "sample_id" not in feature_result:
            # 如果找不到sample_id，手动生成一个
            logging.warning("特征提取结果中未找到sample_id，使用自动生成的ID")
            sample_id = f"sample_{hash(sequence) % 10000:04d}"
        else:
            sample_id = feature_result["sample_id"]
        
        # 2. 初始化预测器
        config = PredictorConfig()
        if model_dir:
            config.model_dir = model_dir
        predictor = ThermostabilityPredictor(config)
        
        # 3. 从特征预测
        prediction = predictor.predict_batch(
            esm2_features_dir=str(temp_dir / "esm2_features"),
            esmc_features_dir=str(temp_dir / "esmc_features"),
            sample_ids=[sample_id],
            return_confidence=return_confidence
        )
        
        # 4. 整理结果
        if "prediction_map" not in prediction or sample_id not in prediction["prediction_map"]:
            raise ValueError(f"预测失败，未找到样本 {sample_id} 的预测结果")
            
        sample_prediction = prediction["prediction_map"][sample_id]
        thermostability = sample_prediction["thermostability"]
        
        result = {
            "sequence": sequence,
            "thermostability": thermostability,
            "interpretation": interpret_thermostability(thermostability)
        }
        
        # 添加置信度（如果有）
        if return_confidence and "confidence" in sample_prediction:
            # 获取原始置信度（标准差）
            raw_confidence = float(sample_prediction["confidence"])
            
            # 归一化置信度到0-1区间
            # 假设最大标准差为20，大于20的视为完全不可信
            max_std = 20.0
            normalized_confidence = max(0.0, 1.0 - min(raw_confidence / max_std, 1.0))
            
            # 保存归一化的置信度和原始标准差
            result["confidence"] = normalized_confidence
            result["std_dev"] = raw_confidence
            
            # 添加置信度解释
            if normalized_confidence >= 0.8:
                result["confidence_level"] = "非常高"
            elif normalized_confidence >= 0.6:
                result["confidence_level"] = "高"
            elif normalized_confidence >= 0.4:
                result["confidence_level"] = "中等"
            elif normalized_confidence >= 0.2:
                result["confidence_level"] = "低"
            else:
                result["confidence_level"] = "非常低"
        
        # 5. 清理临时文件
        if cleanup_features and temp_dir.exists():
            shutil.rmtree(temp_dir)
            
        return result
        
    except Exception as e:
        logging.error(f"预测失败: {str(e)}")
        return {"error": str(e), "sequence": sequence}

def batch_predict(sequences, names=None, with_confidence=True, export_csv=None):
    """批量预测多个蛋白质序列的热稳定性
    
    Args:
        sequences: 蛋白质序列列表
        names: 蛋白质名称列表（可选），如果不提供则使用序列索引
        with_confidence: 是否返回置信度
        export_csv: 导出结果到CSV文件的路径（可选）
        
    Returns:
        预测结果列表
    """
    if names is None:
        names = [f"蛋白质_{i+1}" for i in range(len(sequences))]
    
    results = []
    print(f"\n==== 开始批量预测 {len(sequences)} 个序列的热稳定性 ====")
    
    for i, (name, seq) in enumerate(zip(names, sequences)):
        print(f"\n[{i+1}/{len(sequences)}] 预测 {name} ({len(seq)} aa) 的热稳定性")
        start_time = time.time()
        
        # 进行预测
        result = predict_thermostability(seq, return_confidence=with_confidence)
        
        # 记录执行时间
        elapsed = time.time() - start_time
        
        # 处理结果
        if 'error' in result:
            print(f"预测出错: {result['error']}")
            result_info = {
                'name': name,
                'sequence': seq[:20] + "..." if len(seq) > 20 else seq,
                'error': result['error']
            }
        else:
            print(f"预测热稳定性: {result['thermostability']:.2f}/102")
            if with_confidence and 'confidence' in result:
                print(f"置信度: {result['confidence']:.2f} ({result['confidence_level']})")
                print(f"标准差: {result['std_dev']:.2f}")
            print(f"解释: {result['interpretation']}")
            print(f"耗时: {elapsed:.2f}秒")
            
            result_info = {
                'name': name,
                'sequence': seq[:20] + "..." if len(seq) > 20 else seq,
                'thermostability': result['thermostability'],
                'interpretation': result['interpretation']
            }
            
            if with_confidence and 'confidence' in result:
                result_info['confidence'] = result['confidence']
                result_info['confidence_level'] = result['confidence_level']
                result_info['std_dev'] = result['std_dev']
        
        results.append(result_info)
    
    # 导出到CSV（如果需要）
    if export_csv:
        try:
            df = pd.DataFrame(results)
            df.to_csv(export_csv, index=False, encoding='utf-8-sig')
            print(f"\n预测结果已导出到: {export_csv}")
        except Exception as e:
            print(f"导出CSV失败: {str(e)}")
    
    return results