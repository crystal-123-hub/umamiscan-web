

import pandas as pd
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from atom_feature import convert_to_graph_channel

# 残基到索引的映射
Pep_residue2idx = {
    '[PAD]': 0,
    '[CLS]': 1,
    '[SEP]': 2,
    'A': 3,   # Alanine
    'C': 4,   # Cysteine
    'D': 5,   # Aspartic acid
    'E': 6,   # Glutamic acid
    'F': 7,   # Phenylalanine
    'G': 8,   # Glycine
    'H': 9,   # Histidine
    'I': 10,  # Isoleucine
    'K': 11,  # Lysine
    'L': 12,  # Leucine
    'M': 13,  # Methionine
    'N': 14,  # Asparagine
    'P': 15,  # Proline
    'Q': 16,  # Glutamine
    'R': 17,  # Arginine
    'S': 18,  # Serine
    'T': 19,  # Threonine
    'V': 20,  # Valine
    'W': 21,  # Tryptophan
    'Y': 22,  # Tyrosine
}



# def load_data_from_fasta(file_path):
#     sequences = []
#     labels = []
#     current_seq = ''  # 初始化为空字符串
#
#     with open(file_path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#
#             if line.startswith('>'):  # 遇到新 header，保存上一个序列
#                 if current_seq:  # 如果已经有内容，说明是上一个序列
#                     sequences.append(current_seq)
#                     current_seq = ''  # 清空 current_seq
#
#                 header = line[1:]  # 去掉 '>' 字符
#                 if header.startswith('trAMP') or header.startswith('teAMP'):
#                     labels.append(1)
#                 elif header.startswith('trNEGATIVE') or header.startswith('teNEGATIVE'):
#                     labels.append(0)
#                 else:
#                     raise ValueError(f"Unknown label in header: {header}")
#             else:
#                 current_seq += line  # 累加序列内容
#
#         # 处理最后一个序列
#         if current_seq:
#             sequences.append(current_seq)
#
#     graph_features = [convert_to_graph_channel(seq) for seq in sequences]
#     return graph_features, labels
def load_data_from_fasta(file_path):
    sequences = []
    labels = []
    current_seq = ''  # 初始化为空字符串

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq)
                    current_seq = ''
                label_str = line[1:].strip()
                try:
                    label = int(label_str)
                except ValueError:
                    raise ValueError(f"Header must be an integer label (0 or 1), got: {label_str}")

                if label not in [0, 1]:
                    raise ValueError(f"Label must be 0 or 1, got: {label}")

                labels.append(label)

            else:
                current_seq += line
        if current_seq:
            sequences.append(current_seq)

    # 调用图结构特征提取函数
    graph_features = [convert_to_graph_channel(seq) for seq in sequences]

    return graph_features, labels

class MyDataSet(Data.Dataset):
    def __init__(self, input_esmids, graph_features, labels):
        self.input_esmids = input_esmids
        # self.input_bioids=input_bioids
        self.graph_features = graph_features
        self.labels = labels

    def __len__(self):
        return len(self.input_esmids)

    def __getitem__(self, idx):
        return (
            self.input_esmids[idx].clone().detach().float(),
            # self.input_bioids[idx].clone().detach().float(),
            torch.tensor(self.graph_features[idx], dtype=torch.float),
            torch.tensor(self.labels[idx], dtype=torch.float),
        )


