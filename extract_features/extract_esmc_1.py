import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

def read_csv_and_extract_data(file_path):
    """读取CSV文件并提取所需数据"""
    df = pd.read_csv(file_path)
    return {
        "index": df["index"].tolist(),
        "gene": df["gene"].tolist(),
        "solubility": df["solubility"].tolist(),
        "sequence": df["sequence"].tolist()
    }

def extract_features_with_esmc(file_path, save_path, model_name="esmc_600m"):
    """
    使用ESMC提取蛋白质序列特征（改进版）
    
    改进内容：
    1. 去除嵌入的首尾标识符
    2. 统一填充序列长度到1400
    3. 添加mask区分真实嵌入和填充部分
    """
    # 读取数据
    datasets = read_csv_and_extract_data(file_path)
    indices = datasets["index"]
    sequences = datasets["sequence"]
    solubilities = datasets["solubility"]
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载ESMC模型
    client = ESMC.from_pretrained(model_name).to(device)
    client.eval()
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 单序列处理
    with torch.no_grad():
        for idx, seq, solubility in tqdm(
            zip(indices, sequences, solubilities),
            total=len(sequences),
            desc="Processing sequences"
        ):
            # 验证序列有效性
            if not isinstance(seq, str) or len(seq) == 0:
                print(f"跳过无效序列：索引 {idx}")
                continue
                
            try:
                # 准备蛋白质数据
                protein = ESMProtein(sequence=seq)
                protein_tensor = client.encode(protein).to(device)
                
                # 获取特征
                logits_output = client.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True)
                )
                
                # 提取并处理嵌入特征
                embeddings = logits_output.embeddings
                
                # 去除首尾标识符（CLS和SEP）
                processed_embeddings = embeddings[0][1:-1].cpu().numpy()
                
                # 处理空序列情况
                if processed_embeddings.shape[0] == 0:
                    print(f"警告：序列 {idx} 处理后长度为0，跳过")
                    continue
                
                # 填充/截断到指定长度
                max_length = 1400
                current_len = processed_embeddings.shape[0]
                
                # 处理序列填充/截断
                if current_len < max_length:
                    pad_len = max_length - current_len
                    padded_residue = np.pad(processed_embeddings, 
                                          ((0, pad_len), (0, 0)), 
                                          mode='constant')
                    mask = np.concatenate([
                        np.ones(current_len, dtype=int),
                        np.zeros(pad_len, dtype=int)
                    ])
                elif current_len > max_length:
                    padded_residue = processed_embeddings[:max_length, :]
                    mask = np.ones(max_length, dtype=int)
                else:
                    padded_residue = processed_embeddings
                    mask = np.ones(max_length, dtype=int)
                
                # 构建特征字典
                feature_data = {
                    "index": idx,
                    "sequence": seq,
                    "solubility": solubility,
                    "protein_representation": embeddings[0, 0].cpu().numpy(),  # CLS token
                    "residue_representation": padded_residue,
                    "mask": mask
                }
                
                # 保存为.npy文件
                np.save(
                    os.path.join(save_path, f"{idx}_features.npy"),
                    feature_data
                )
            except Exception as e:
                print(f"处理序列 {idx} 时出错：{str(e)}")
                continue

if __name__ == "__main__":
    extract_features_with_esmc(
        file_path="eSol_train.csv",
        save_path="./esmc_features_train/",
        model_name="esmc_600m"
    )
    extract_features_with_esmc(
        file_path="eSol_test.csv",
        save_path="./esmc_features_test/",
        model_name="esmc_600m"
    )
    extract_features_with_esmc(
        file_path="S.cerevisiae_test.csv",
        save_path="./esmc_features_cerevisiae/",
        model_name="esmc_600m"
    )