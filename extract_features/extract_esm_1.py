import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig

def read_csv_and_extract_data(file_path):
    """读取CSV文件并提取所需数据"""
    df = pd.read_csv(file_path)
    return {
        "index": df["index"].tolist(),
        "gene": df["gene"].tolist(),
        "solubility": df["solubility"].tolist(),
        "sequence": df["sequence"].tolist()
    }

def extract_features_with_esm2(file_path, save_path, model_name="facebook/esm2_t33_650M_UR50D"):
    """
    使用ESM-2提取蛋白质序列特征 (基于transformers库)
    
    功能:
    1. 提取残基级特征
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
    print(f"使用设备: {device}")
    
    # 加载ESM-2模型
    print(f"正在加载ESM-2模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
    # 关闭dropout以确保推理结果的确定性
    config.hidden_dropout = 0.
    config.hidden_dropout_prob = 0.
    config.attention_dropout = 0.
    config.attention_probs_dropout_prob = 0.
    
    # 加载模型并设置为评估模式
    encoder = AutoModel.from_pretrained(model_name, config=config).to(device).eval()
    print("ESM2 模型加载完成")
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 单序列处理函数
    def seq_encode(seq):
        # 在氨基酸之间添加空格
        spaced_seq = " ".join(list(seq))
        # 编码序列
        inputs = tokenizer.encode_plus(
            spaced_seq, 
            return_tensors=None, 
            add_special_tokens=True,
            padding=True,
            truncation=True
        )
        # 转换为tensor并移至设备
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long).unsqueeze(0).to(device)
        
        # 前向传播
        with torch.no_grad():
            outputs = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        
        # 提取最后一层隐藏状态
        last_hidden_states = outputs[0]
        # 提取有效token的嵌入 (跳过首尾特殊标记)
        encoded_seq = last_hidden_states[0, inputs['attention_mask'][0].bool()][1:-1]
        return encoded_seq
    
    # 逐序列处理
    with torch.no_grad():
        for idx, seq, solubility in tqdm(
            zip(indices, sequences, solubilities),
            total=len(sequences),
            desc="处理序列"
        ):
            # 验证序列有效性
            if not isinstance(seq, str) or len(seq) == 0:
                print(f"跳过无效序列：索引 {idx}")
                continue
                
            try:
                # 使用优化后的编码函数获取残基嵌入
                encoded_residues = seq_encode(seq)
                
                # 从编码中获取残基表示
                processed_embeddings = encoded_residues.cpu().numpy()
                
                # 计算蛋白质整体表示 (使用平均池化)
                protein_representation = processed_embeddings.mean(axis=0)
                
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
                    "protein_representation": protein_representation,  # 平均池化的序列表示
                    "residue_representation": padded_residue,  # 每个残基的表示
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
    # 使用ESM2提取特征
    extract_features_with_esm2(
        file_path="eSol_train.csv",
        save_path="./esm2_features_train/",
        model_name="/HOME/scz0brz/run/model/esm2_t33_650M_UR50D"  # 使用与SaProt相当规模的模型
    )
    extract_features_with_esm2(
        file_path="eSol_test.csv",
        save_path="./esm2_features_test/", 
        model_name="/HOME/scz0brz/run/model/esm2_t33_650M_UR50D"
    )
    extract_features_with_esm2(
        file_path="S.cerevisiae_test.csv",
        save_path="./esm2_features_cerevisiae/", 
        model_name="/HOME/scz0brz/run/model/esm2_t33_650M_UR50D"
    )