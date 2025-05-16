"""
蛋白质预测模型管理器 - 用于优化批量预测性能
"""
import os
import time
import logging
import pandas as pd
from pathlib import Path
import shutil
import torch
import uuid
import numpy as np

logger = logging.getLogger("PredictorManager")

class ProteinPredictorManager:
    """蛋白质预测模型管理器，负责加载和管理所有预测模型，优化批量预测性能"""
    
    def __init__(self, 
                toxicity_model_dir=None,
                thermostability_model_dir=None, 
                solubility_model_dir=None,
                use_toxicity=True,
                use_thermostability=True,
                use_solubility=True,
                feature_manager_instance=None):
        """初始化预测管理器
        
        Args:
            toxicity_model_dir: 毒性模型目录，None则使用默认
            thermostability_model_dir: 热稳定性模型目录，None则使用默认
            solubility_model_dir: 溶解性模型目录，None则使用默认
            use_toxicity: 是否加载毒性预测模型
            use_thermostability: 是否加载热稳定性预测模型
            use_solubility: 是否加载溶解性预测模型
            feature_manager_instance: 传入的 FeatureManager 实例 (可选)
        """
        self.feature_manager = feature_manager_instance
        self.toxicity_predictor = None
        self.thermostability_predictor = None
        self.solubility_predictor = None
        
        # 记录开始时间，用于计算模型加载时间
        start_time = time.time()
        
        # 加载毒性预测模型
        if use_toxicity:
            from .toxicity.config import PredictorConfig as ToxicityConfig
            from .toxicity.predictor import ToxicityPredictor
            
            tox_config = ToxicityConfig()
            if toxicity_model_dir:
                tox_config.model_dir = toxicity_model_dir
                
            logger.info(f"加载毒性预测模型...")
            self.toxicity_predictor = ToxicityPredictor(tox_config)
        
        # 加载热稳定性预测模型
        if use_thermostability:
            from .thermostability.config import PredictorConfig as ThermostabilityConfig
            from .thermostability.predictor import ThermostabilityPredictor
            
            thermo_config = ThermostabilityConfig()
            if thermostability_model_dir:
                thermo_config.model_dir = thermostability_model_dir
                
            logger.info(f"加载热稳定性预测模型...")
            self.thermostability_predictor = ThermostabilityPredictor(thermo_config)
        
        # 加载溶解性预测模型
        if use_solubility:
            from .solubility.config import PredictorConfig as SolubilityConfig
            from .solubility.predictor import SolubilityPredictor
            
            solubility_config = SolubilityConfig()
            if solubility_model_dir:
                solubility_config.model_dir = solubility_model_dir
                
            logger.info(f"加载溶解性预测模型...")
            self.solubility_predictor = SolubilityPredictor(solubility_config)
        
        # 计算并记录模型加载时间
        load_time = time.time() - start_time
        logger.info(f"所有预测模型加载完成，耗时: {load_time:.2f}秒")
    
    def predict_all(self, sequence, return_confidence=False, cleanup_features=True):
        """同时预测蛋白质的毒性、热稳定性和溶解性
        
        Args:
            sequence: 蛋白质序列
            return_confidence: 是否返回预测置信度
            cleanup_features: 完成后是否删除临时特征文件
            
        Returns:
            dict: 包含所有特性预测结果的字典
        """
        current_run_temp_dir = None
        try:
            if not self.feature_manager:
                logger.error("FeatureManager instance is not available in ProteinPredictorManager.")
                raise RuntimeError("FeatureManager is required by ProteinPredictorManager but not provided.")

            sample_id = uuid.uuid4().hex

            # Create a unique base temporary directory for this prediction run
            # Using a "predict_temp_<random_id>" structure under a common base
            base_temp_storage = Path("./temp_protein_predict_runs") 
            base_temp_storage.mkdir(exist_ok=True)
            
            current_run_temp_dir_name = f"predict_run_{sample_id}"
            current_run_temp_dir = base_temp_storage / current_run_temp_dir_name
            current_run_temp_dir.mkdir(parents=True, exist_ok=False) 

            esm2_features_dir = current_run_temp_dir / "esm2_features"
            esmc_features_dir = current_run_temp_dir / "esmc_features"
            esm2_features_dir.mkdir()
            esmc_features_dir.mkdir()

            # Convert to string paths for sub-predictors
            str_esm2_features_dir = str(esm2_features_dir)
            str_esmc_features_dir = str(esmc_features_dir)

            # Extract and save ESM2 features
            if 'esm2' in self.feature_manager.extractors:
                try:
                    logger.info(f"[{sample_id}] Attempting to extract ESM2 features for sequence: {sequence[:30]}...")
                    esm2_data = self.feature_manager.extract_features(sequence, "esm2")
                    logger.info(f"[{sample_id}] ESM2 data extracted: {list(esm2_data.keys()) if isinstance(esm2_data, dict) else 'Not a dict'}")
                    # Ensure esm2_data is a dictionary before trying to save
                    if isinstance(esm2_data, dict):
                        esm2_file_path = esm2_features_dir / f"{sample_id}_features.npy" # Changed extension and filename part
                        np.save(esm2_file_path, esm2_data) # Save the whole dictionary using np.save
                        logger.info(f"[{sample_id}] ESM2 feature dictionary saved to: {esm2_file_path}. Exists: {esm2_file_path.exists()}")
                    else:
                        logger.error(f"[{sample_id}] ESM2 feature extraction did not return a dictionary. Skipping save.")
                        esm2_features_available = False
                except Exception as e_feat:
                    logger.error(f"[{sample_id}] Error extracting/saving ESM2 features: {e_feat}", exc_info=True)
                    esm2_features_available = False
            else:
                logger.warning(f"[{sample_id}] ESM2 extractor not found in feature_manager.")
                esm2_features_available = False

            # Extract and save ESMC features
            if 'esmc' in self.feature_manager.extractors:
                try:
                    logger.info(f"[{sample_id}] Attempting to extract ESMC features for sequence: {sequence[:30]}...")
                    esmc_data = self.feature_manager.extract_features(sequence, "esmc")
                    logger.info(f"[{sample_id}] ESMC data extracted: {list(esmc_data.keys()) if isinstance(esmc_data, dict) else 'Not a dict'}")
                    # Ensure esmc_data is a dictionary
                    if isinstance(esmc_data, dict):
                        esmc_file_path = esmc_features_dir / f"{sample_id}_features.npy" # Changed extension and filename part
                        np.save(esmc_file_path, esmc_data) # Save the whole dictionary using np.save
                        logger.info(f"[{sample_id}] ESMC feature dictionary saved to: {esmc_file_path}. Exists: {esmc_file_path.exists()}")
                    else:
                        logger.error(f"[{sample_id}] ESMC feature extraction did not return a dictionary. Skipping save.")
                        esmc_features_available = False
                except Exception as e_feat:
                    logger.error(f"[{sample_id}] Error extracting/saving ESMC features: {e_feat}", exc_info=True)
                    esmc_features_available = False
            else:
                logger.warning(f"[{sample_id}] ESMC extractor not found in feature_manager.")
                esmc_features_available = False
            
            # 准备结果字典
            result = {"sequence": sequence}
            logger.debug(f"[{sample_id}] Initial result dict: {result}")

            # 毒性预测
            if self.toxicity_predictor:
                logger.info(f"[{sample_id}] Calling ToxicityPredictor with esm2_dir='{str_esm2_features_dir}', esmc_dir='{str_esmc_features_dir}', ids=['{sample_id}']")
                tox_result = self.toxicity_predictor.predict_batch(
                    esm2_features_dir=str_esm2_features_dir,
                    esmc_features_dir=str_esmc_features_dir,
                    sample_ids=[sample_id],
                    return_confidence=return_confidence
                )
                logger.info(f"[{sample_id}] Toxicity_predictor raw result: {tox_result}")
                if tox_result and "prediction_map" in tox_result:
                    logger.info(f"[{sample_id}] Toxicity prediction_map keys: {list(tox_result['prediction_map'].keys())}. Checking for sample_id: {sample_id}")
                    if sample_id in tox_result["prediction_map"]:
                        tox_prediction = tox_result["prediction_map"][sample_id]
                        toxicity_prob = tox_prediction["toxicity_prob"]
                        is_toxic = tox_prediction["is_toxic"]
                        
                        from .toxicity.utils import interpret_toxicity
                        interpretation = interpret_toxicity(toxicity_prob)
                        
                        toxicity_result = {
                            "toxicity_probability": toxicity_prob,
                            "is_toxic": is_toxic,
                            "risk_level": interpretation["risk_level"],
                            "description": interpretation["description"],
                            "classification": interpretation["classification"]
                        }
                        
                        if return_confidence and "confidence" in tox_prediction:
                            toxicity_result["confidence"] = tox_prediction["confidence"]
                            
                        result["toxicity"] = toxicity_result
            
            # 热稳定性预测
            if self.thermostability_predictor:
                logger.info(f"[{sample_id}] Calling ThermostabilityPredictor with esm2_dir='{str_esm2_features_dir}', esmc_dir='{str_esmc_features_dir}', ids=['{sample_id}']")
                thermo_result = self.thermostability_predictor.predict_batch(
                    esm2_features_dir=str_esm2_features_dir,
                    esmc_features_dir=str_esmc_features_dir,
                    sample_ids=[sample_id],
                    return_confidence=return_confidence
                )
                logger.info(f"[{sample_id}] Thermostability_predictor raw result: {thermo_result}")
                if thermo_result and "prediction_map" in thermo_result:
                    logger.info(f"[{sample_id}] Thermostability prediction_map keys: {list(thermo_result['prediction_map'].keys())}. Checking for sample_id: {sample_id}")
                    if sample_id in thermo_result["prediction_map"]:
                        thermo_prediction = thermo_result["prediction_map"][sample_id]
                        thermostability = thermo_prediction["thermostability"]
                        
                        from .thermostability.utils import interpret_thermostability
                        thermo_interpretation = interpret_thermostability(thermostability)
                        
                        thermostability_result = {
                            "thermostability": thermostability,
                            "interpretation": thermo_interpretation
                        }
                        
                        if return_confidence and "confidence" in thermo_prediction:
                            thermostability_result["confidence"] = thermo_prediction["confidence"]
                            
                        result["thermostability"] = thermostability_result
            
            # 溶解性预测
            if self.solubility_predictor:
                logger.info(f"[{sample_id}] Calling SolubilityPredictor with esm2_dir='{str_esm2_features_dir}', esmc_dir='{str_esmc_features_dir}', ids=['{sample_id}']")
                sol_result = self.solubility_predictor.predict_batch(
                    esm2_features_dir=str_esm2_features_dir,
                    esmc_features_dir=str_esmc_features_dir,
                    sample_ids=[sample_id],
                    return_confidence=return_confidence
                )
                logger.info(f"[{sample_id}] Solubility_predictor raw result: {sol_result}")
                if sol_result and "prediction_map" in sol_result:
                    logger.info(f"[{sample_id}] Solubility prediction_map keys: {list(sol_result['prediction_map'].keys())}. Checking for sample_id: {sample_id}")
                    if sample_id in sol_result["prediction_map"]:
                        sol_prediction = sol_result["prediction_map"][sample_id]
                        solubility = sol_prediction["solubility"]
                        
                        from .solubility.utils import interpret_solubility
                        sol_interpretation = interpret_solubility(solubility)
                        
                        solubility_result = {
                            "solubility": solubility,
                            "interpretation": sol_interpretation
                        }
                        
                        if return_confidence and "confidence" in sol_prediction:
                            solubility_result["confidence"] = sol_prediction["confidence"]
                            
                        result["solubility"] = solubility_result
            
            logger.info(f"Predict_all result for sample_id {sample_id}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"预测失败 (predict_all): {str(e)}", exc_info=True)
            return {"error": str(e), "sequence": sequence}
        finally:
            if cleanup_features and current_run_temp_dir and current_run_temp_dir.exists():
                try:
                    shutil.rmtree(current_run_temp_dir)
                    logger.debug(f"Successfully cleaned up temp directory: {current_run_temp_dir}")
                except Exception as e_cleanup:
                    logger.error(f"Failed to cleanup temp directory {current_run_temp_dir}: {e_cleanup}", exc_info=True)
    
    def batch_predict_all(self, sequences, names=None, with_confidence=True, export_csv=None):
        """批量同时预测多个蛋白质序列的所有特性
        
        Args:
            sequences: 蛋白质序列列表
            names: 蛋白质名称列表（可选）
            with_confidence: 是否返回置信度
            export_csv: 导出结果到CSV文件的路径（可选）
            
        Returns:
            list: 预测结果列表
        """
        if names is None:
            names = [f"蛋白质_{i+1}" for i in range(len(sequences))]
        
        results = []
        print(f"\n==== 开始批量预测 {len(sequences)} 个序列的蛋白质特性 ====")
        
        for i, (name, seq) in enumerate(zip(names, sequences)):
            print(f"\n[{i+1}/{len(sequences)}] 预测 {name} ({len(seq)} aa)")
            start_time = time.time()
            
            # 进行预测
            result = self.predict_all(seq, return_confidence=with_confidence)
            
            # 记录执行时间
            elapsed = time.time() - start_time
            
            # 处理结果
            if 'error' in result:
                print(f"  ❌ 预测失败: {result['error']} (用时 {elapsed:.1f}s)")
                result_info = {
                    "name": name, 
                    "length": len(seq), 
                    "error": result["error"],
                    "time": elapsed
                }
            else:
                result_info = {
                    "name": name,
                    "length": len(seq),
                    "time": elapsed
                }
                
                # 处理毒性结果
                if "toxicity" in result:
                    tox = result["toxicity"]
                    print(f"  - 毒性: {tox['toxicity_probability']:.3f} - {tox['risk_level']} - {tox['classification']}")
                    
                    result_info.update({
                        "toxicity": tox["toxicity_probability"],
                        "risk_level": tox["risk_level"],
                        "classification": tox["classification"]
                    })
                    
                    if with_confidence and "confidence" in tox:
                        result_info["toxicity_confidence"] = tox["confidence"]
                
                # 处理热稳定性结果
                if "thermostability" in result:
                    thermo = result["thermostability"]
                    print(f"  - 热稳定性: {thermo['thermostability']:.1f} - {thermo['interpretation']}")
                    
                    result_info.update({
                        "thermostability": thermo["thermostability"],
                        "thermostability_interpretation": thermo["interpretation"]
                    })
                    
                    if with_confidence and "confidence" in thermo:
                        result_info["thermostability_confidence"] = thermo["confidence"]
                
                # 处理溶解性结果
                if "solubility" in result:
                    sol = result["solubility"]
                    print(f"  - 溶解性: {sol['solubility']:.3f} - {sol['interpretation']}")
                    
                    result_info.update({
                        "solubility": sol["solubility"],
                        "solubility_interpretation": sol["interpretation"]
                    })
                    
                    if with_confidence and "confidence" in sol:
                        result_info["solubility_confidence"] = sol["confidence"]
                
                print(f"  ✅ 预测完成 (用时 {elapsed:.1f}s)")
            
            results.append(result_info)
        
        # 导出到CSV（如果需要）
        if export_csv:
            try:
                df = pd.DataFrame(results)
                df.to_csv(export_csv, index=False, encoding='utf-8-sig')
                print(f"\n✅ 结果已导出到: {export_csv}")
            except Exception as e:
                print(f"\n❌ 导出结果失败: {str(e)}")
        
        return results

# 导出管理器类
__all__ = ["ProteinPredictorManager"]