import torch.nn as nn
import torch.nn.functional as F
import math
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json
from sklearn.preprocessing import MinMaxScaler


def prepare_dataset_and_scaler(input_file, output_csv, scaler_json_path, seq_length=9):
    # 1. 读取原始CSV文件
    df = pd.read_csv(input_file)

    # 2. 定义特征列（区分图片和数值）
    image_cols = ['front_image', 'back_image', 'left_image']
    numeric_input_cols = [
        'global_x', 'global_y', 'global_z',
        'velocity_x', 'velocity_y', 'velocity_z',
        'steer', 'acceleration_x', 'acceleration_y', 'acceleration_z'
    ]
    target_cols = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'steer']

    # 组合所有输入列用于后续序列构建
    input_cols = image_cols + numeric_input_cols
    # 确定所有需要归一化的数值列
    all_numeric = sorted(list(set(numeric_input_cols + target_cols)))

    # 3. 数据归一化 (仅针对数值列)
    scaler = MinMaxScaler()
    df_norm = df.copy()
    # 关键修改：只对数值列执行 fit_transform
    df_norm[all_numeric] = scaler.fit_transform(df[all_numeric])

    # 4. 保存归一化参数到JSON文件
    scaler_params = {}
    for i, col in enumerate(all_numeric):
        scaler_params[col] = {
            'min': float(scaler.data_min_[i]),
            'max': float(scaler.data_max_[i])
        }
    with open(scaler_json_path, 'w') as f:
        json.dump(scaler_params, f, indent=4)

    # 5. 构建历史序列数据
    data_list = []
    for i in range(len(df_norm)):
        row_dict = {'target_timestamp': df.iloc[i]['timestamp']}
        # 获取历史seq_length个时间点的索引 (不足时用起始行填充)
        indices = [max(0, i - (seq_length - 1 - step)) for step in range(seq_length)]

        # 为每个特征列构建历史序列
        for col in input_cols:
            # 这里 history_vals 会根据列类型自动包含字符串或归一化后的浮点数
            history_vals = df_norm.iloc[indices][col].tolist()
            row_dict[f'{col}_history'] = json.dumps(history_vals)

        # 保存目标值 (已经是归一化后的数值)
        for col in target_cols:
            row_dict[f'target_{col}'] = df_norm.iloc[i][col]
        data_list.append(row_dict)

    # 6. 保存处理后的数据到新的CSV文件
    pd.DataFrame(data_list).to_csv(output_csv, index=False)
    return scaler_params

def inverse_transform(normalized_data, target_names, scaler_params):
    """将归一化的数据还原为原始值"""
    real_values = []
    for i, col in enumerate(target_names):
        p = scaler_params[col]
        actual_val = normalized_data[i] * (p['max'] - p['min']) + p['min']
        real_values.append(actual_val)
    return real_values