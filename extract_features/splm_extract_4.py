import yaml
import os
import numpy as np
import torch
from tqdm import tqdm
from utils import load_configs, load_checkpoints_only
from model import SequenceRepresentation
import pandas as pd

def read_csv_and_extract_data(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 提取需要的列
    datasets = {
        "index": df["index"].tolist(),
        "gene": df["gene"].tolist(),
        "solubility": df["solubility"].tolist(),
        "sequence": df["sequence"].tolist()
    }
    
    return datasets

def extract_features_from_csv(file_path, config_path, checkpoint_path, save_path, device='cuda'):
    """从CSV文件中提取蛋白质序列特征并保存"""
    # 读取CSV文件中的数据
    datasets = read_csv_and_extract_data(file_path)

    # 提取数据
    sequences = datasets["sequence"]
    solubilities = datasets["solubility"]
    indices = datasets["index"]

    # 加载配置文件
    with open(config_path) as file:
        dict_config = yaml.full_load(file)
    configs = load_configs(dict_config)

    # 创建模型
    model = SequenceRepresentation(logging=None, configs=configs)
    model.to(device)
    load_checkpoints_only(checkpoint_path, model)

    # 将序列转换为适合模型输入的格式
    esm2_seq = [(range(len(sequences)), str(sequences[i])) for i in range(len(sequences))]
    batch_labels, batch_strs, batch_tokens = model.batch_converter(esm2_seq)

    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    # 使用进度条处理数据
    for i in tqdm(range(len(sequences)), desc="提取特征", unit="序列"):
        # 获取单个序列的表示
        token = batch_tokens[i].unsqueeze(0).to(device)
        
        # 填充/截断到1400长度
        if token.size(1) < 1400:
            padding = torch.ones((1, 1400 - token.size(1)), dtype=token.dtype, device=device) * model.alphabet.padding_idx
            token = torch.cat([token, padding], dim=1)
        elif token.size(1) > 1400:
            token = token[:, :1400]  # 截断过长序列

        # 获取蛋白质表示、残基表示和掩码
        protein_representation, residue_representation, mask = model(token)
        
        # 打印原始维度
        print(f"原始维度 - protein: {protein_representation.shape}, residue: {residue_representation.shape}, mask: {mask.shape}")
        
        # 移除批次维度(第一维)
        protein_rep = protein_representation.squeeze(0).cpu().numpy()  # 从(1, 1280)变为(1280)
        residue_rep = residue_representation.squeeze(0).cpu().numpy()  # 从(1, 1400, 1280)变为(1400, 1280)
        mask_data = mask.squeeze(0).cpu().numpy()  # 从(1, 1400)变为(1400)
        
        # 打印处理后的维度
        print(f"处理后维度 - protein: {protein_rep.shape}, residue: {residue_rep.shape}, mask: {mask_data.shape}")
        
        # 获取当前序列的索引和序列本身
        index = indices[i]
        sequence = sequences[i]
        solubility = solubilities[i]
        
        # 打包数据
        feature_data = {
            "index": index,
            "sequence": sequence,
            "solubility": solubility,
            "protein_representation": protein_rep,
            "residue_representation": residue_rep,
            "mask": mask_data
        }

        # 保存为 .npy 文件
        np.save(os.path.join(save_path, f"{index}_features.npy"), feature_data)

    print("特征已保存为带索引和序列的.npy文件。")

# 示例调用
if __name__ == '__main__':
    file_path = "eSol_train.csv"
    config_path = "./configs/representation_config.yaml"
    checkpoint_path = "/HOME/scz0brz/run/AA_solubility/model/checkpoint_0520000.pth"
    save_path = "./splm_features_train_1/"  # 使用新目录避免覆盖原始数据
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    extract_features_from_csv(file_path, config_path, checkpoint_path, save_path, device)

    file_path = "eSol_test.csv"
    config_path = "./configs/representation_config.yaml"
    checkpoint_path = "/HOME/scz0brz/run/AA_solubility/model/checkpoint_0520000.pth"
    save_path = "./splm_features_test_1/"  # 使用新目录避免覆盖原始数据
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    extract_features_from_csv(file_path, config_path, checkpoint_path, save_path, device)

    file_path = "S.cerevisiae_test.csv"
    config_path = "./configs/representation_config.yaml"
    checkpoint_path = "/HOME/scz0brz/run/AA_solubility/model/checkpoint_0520000.pth"
    save_path = "./splm_features_cerevisiae_1/"  # 使用新目录避免覆盖原始数据
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    extract_features_from_csv(file_path, config_path, checkpoint_path, save_path, device)