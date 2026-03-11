import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json
from sklearn.preprocessing import MinMaxScaler

# def prepare_dataset_and_scaler(input_file, output_csv, scaler_json_path, seq_length=9):
#     # 1. 读取原始CSV文件
#     df = pd.read_csv(input_file)
#
#     # 2. 定义特征列（区分图片和数值）
#     image_cols = ['front_image', 'back_image', 'left_image']
#     numeric_input_cols = [
#         'global_x', 'global_y', 'global_z',
#         'velocity_x', 'velocity_y', 'velocity_z',
#         'steer', 'acceleration_x', 'acceleration_y', 'acceleration_z'
#     ]
#     target_cols = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'steer']
#
#     # 组合所有输入列用于后续序列构建
#     input_cols = image_cols + numeric_input_cols
#     # 确定所有需要归一化的数值列
#     all_numeric = sorted(list(set(numeric_input_cols + target_cols)))
#
#     # 3. 数据归一化 (仅针对数值列)
#     scaler = MinMaxScaler()
#     df_norm = df.copy()
#     # 关键修改：只对数值列执行 fit_transform
#     df_norm[all_numeric] = scaler.fit_transform(df[all_numeric])
#
#     # 4. 保存归一化参数到JSON文件
#     scaler_params = {}
#     for i, col in enumerate(all_numeric):
#         scaler_params[col] = {
#             'min': float(scaler.data_min_[i]),
#             'max': float(scaler.data_max_[i])
#         }
#     with open(scaler_json_path, 'w') as f:
#         json.dump(scaler_params, f, indent=4)
#
#     # 5. 构建历史序列数据
#     data_list = []
#     for i in range(len(df_norm)):
#         row_dict = {'target_timestamp': df.iloc[i]['timestamp']}
#         # 获取历史seq_length个时间点的索引 (不足时用起始行填充)
#         indices = [max(0, i - (seq_length - 1 - step)) for step in range(seq_length)]
#
#         # 为每个特征列构建历史序列
#         for col in input_cols:
#             # 这里 history_vals 会根据列类型自动包含字符串或归一化后的浮点数
#             history_vals = df_norm.iloc[indices][col].tolist()
#             row_dict[f'{col}_history'] = json.dumps(history_vals)
#
#         # 保存目标值 (已经是归一化后的数值)
#         for col in target_cols:
#             row_dict[f'target_{col}'] = df_norm.iloc[i][col]
#         data_list.append(row_dict)
#
#     # 6. 保存处理后的数据到新的CSV文件
#     pd.DataFrame(data_list).to_csv(output_csv, index=False)
#     return scaler_params
"""
上面的是预测加速度的不是速度
"""

# def prepare_dataset_and_scaler(input_file, output_csv, scaler_json_path, seq_length=9):
#     # 1. 读取原始CSV文件
#     df = pd.read_csv(input_file)

#     # 2. 定义特征列（区分图片和数值）
#     image_cols = ['front_image', 'back_image', 'left_image']
#     numeric_input_cols = [
#         'global_x', 'global_y', 'global_z',
#         'velocity_x', 'velocity_y', 'velocity_z',
#         'steer', 'acceleration_x', 'acceleration_y', 'acceleration_z'
#     ]
#     # 修改目标列为速度标量和转角
#     target_cols = ['velocity', 'steer']

#     # 计算速度标量
#     df['velocity'] = (df['velocity_x']**2 + df['velocity_y']**2 + df['velocity_z']**2)**0.5

#     # 组合所有输入列用于后续序列构建
#     input_cols = image_cols + numeric_input_cols
#     all_numeric = sorted(list(set(numeric_input_cols + target_cols)))

#     # 3. 数据归一化 (仅针对数值列)
#     scaler = MinMaxScaler()
#     df_norm = df.copy()
#     df_norm[all_numeric] = scaler.fit_transform(df[all_numeric])

#     # 4. 保存归一化参数到JSON文件
#     scaler_params = {}
#     for i, col in enumerate(all_numeric):
#         scaler_params[col] = {
#             'min': float(scaler.data_min_[i]),
#             'max': float(scaler.data_max_[i])
#         }
#     with open(scaler_json_path, 'w') as f:
#         json.dump(scaler_params, f, indent=4)

#     # 5. 构建历史序列数据
#     data_list = []
#     for i in range(len(df_norm)):
#         row_dict = {'target_timestamp': df.iloc[i]['timestamp']}
#         indices = [max(0, i - (seq_length - 1 - step)) for step in range(seq_length)]

#         # 为每个特征列构建历史序列
#         for col in input_cols:
#             history_vals = df_norm.iloc[indices][col].tolist()
#             row_dict[f'{col}_history'] = json.dumps(history_vals)

#         # 保存目标值 (速度标量和转角)
#         for col in target_cols:
#             row_dict[f'target_{col}'] = df_norm.iloc[i][col]
#         data_list.append(row_dict)

#     # 6. 保存处理后的数据到新的CSV文件
#     pd.DataFrame(data_list).to_csv(output_csv, index=False)
#     return scaler_params

def inverse_transform(normalized_data, target_names, scaler_params):
    """将归一化的数据还原为原始值"""
    real_values = []
    for i, col in enumerate(target_names):
        p = scaler_params[col]
        actual_val = normalized_data[i] * (p['max'] - p['min']) + p['min']
        real_values.append(actual_val)
    return real_values

"""
上面是针对单一csv文件的速度处理
"""

def prepare_multiple_datasets_and_scaler(input_files, output_csv, scaler_json_path, seq_length=9):
    """
    汇总多个 CSV 文件并处理，确保不同文件的首尾不被当作连续帧处理。
    
    Args:
        input_files (list): 包含多个 CSV 文件路径的列表。
        output_csv (str): 输出处理后的 CSV 文件路径。
        scaler_json_path (str): 保存归一化参数的 JSON 文件路径。
        seq_length (int): 历史序列长度。
    """
    # 1. 读取所有 CSV 文件并合并
    dfs = []
    for file in input_files:
        df = pd.read_csv(file)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)

    # 2. 添加文件边界标记
    combined_df['file_boundary'] = 0
    start_idx = 0
    for df in dfs:
        end_idx = start_idx + len(df)
        combined_df.loc[start_idx:end_idx - 1, 'file_boundary'] = 1
        start_idx = end_idx

    # 3. 定义特征列（区分图片和数值）
    image_cols = ['front_image', 'back_image', 'left_image']
    numeric_input_cols = [
        'global_x', 'global_y', 'global_z',
        'velocity_x', 'velocity_y', 'velocity_z',
        'steer', 'acceleration_x', 'acceleration_y', 'acceleration_z'
    ]
    target_cols = ['velocity', 'steer']

    # 计算速度标量
    combined_df['velocity'] = (combined_df['velocity_x']**2 + combined_df['velocity_y']**2 + combined_df['velocity_z']**2)**0.5

    # 4. 数据归一化 (仅针对数值列)
    scaler = MinMaxScaler()
    all_numeric = sorted(list(set(numeric_input_cols + target_cols)))
    combined_df_norm = combined_df.copy()
    combined_df_norm[all_numeric] = scaler.fit_transform(combined_df[all_numeric])

    # 5. 保存归一化参数到 JSON 文件
    scaler_params = {}
    for i, col in enumerate(all_numeric):
        scaler_params[col] = {
            'min': float(scaler.data_min_[i]),
            'max': float(scaler.data_max_[i])
        }
    with open(scaler_json_path, 'w') as f:
        json.dump(scaler_params, f, indent=4)

    # 6. 构建历史序列数据
    data_list = []
    for i in range(len(combined_df_norm)):
        row_dict = {'target_timestamp': combined_df.iloc[i]['timestamp']}
        indices = []

        # 确保历史序列不跨越文件边界
        for step in range(seq_length):
            idx = i - (seq_length - 1 - step)
            if idx < 0 or combined_df.loc[idx, 'file_boundary'] == 0:
                idx = i  # 如果超出边界，用当前帧填充
            indices.append(idx)

        # 为每个特征列构建历史序列
        for col in image_cols + numeric_input_cols:
            history_vals = combined_df_norm.iloc[indices][col].tolist()
            row_dict[f'{col}_history'] = json.dumps(history_vals)

        # 保存目标值 (速度标量和转角)
        for col in target_cols:
            row_dict[f'target_{col}'] = combined_df_norm.iloc[i][col]
        data_list.append(row_dict)

    # 7. 保存处理后的数据到新的 CSV 文件
    pd.DataFrame(data_list).to_csv(output_csv, index=False)
    return scaler_params

# ==================== 1. 适配新格式的 Dataset ====================
class ProcessedDrivingDataset(Dataset):
    """
    专门解析带有 JSON history 列的端到端驾驶数据集
    - 图像流输入: t-2, t-1, t (保留当前帧)
    - 状态流输入: 过去 8 帧 (剔除当前帧，防止数据泄露)
    """

    def __init__(self, csv_file, root_dir="", transform=None):
        self.data_df = pd.read_csv(csv_file)
        self.root_dir = root_dir

        # 定义需要送入 LSTM 的 10 个数值特征列
        self.numeric_cols = [
            'global_x_history', 'global_y_history', 'global_z_history',
            'velocity_x_history', 'velocity_y_history', 'velocity_z_history',
            'steer_history',
            'acceleration_x_history', 'acceleration_y_history', 'acceleration_z_history'
        ]

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]

        # ---------------- A. 视觉数据 (取最后3帧: t-2, t-1, t) ----------------
        front_images = json.loads(row['front_image_history'])
        # 截取最后三张，当前帧是 front_images[-1]
        img_paths = [front_images[-3], front_images[-2], front_images[-1]]

        images = []
        for path in img_paths:
            full_path = os.path.join(self.root_dir, path) if self.root_dir else path
            img = Image.open(full_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)

        img_t_minus_2, img_t_minus_1, img_t = images

        # ---------------- B. 状态历史数据 (去掉当前帧 t) ----------------
        state_features = []
        for col in self.numeric_cols:
            # 解析 json list，原始长度为 9
            val_list = json.loads(row[col])

            # 【核心修改点】：使用 [:-1] 剔除掉列表的最后一个元素（即当前帧t的数据）
            # 这样 val_list 长度变为 8
            state_features.append(val_list[:-1])

        # state_features: 10个特征 x 8帧 -> 转置为 8 x 10
        state_seq = np.array(state_features, dtype=np.float32).T
        state_seq_tensor = torch.tensor(state_seq)

        # ---------------- C. 目标标签 (当前帧 t 的真实动作) ----------------
        # target_tensor = torch.tensor([
        #     row['target_acceleration_x'],
        #     row['target_acceleration_y'],
        #     row['target_acceleration_z'],
        #     row['target_steer']
        # ], dtype=torch.float32)
        target_tensor = torch.tensor([
            row['target_velocity'],  # 速度标量
            row['target_steer']      # 转角
        ], dtype=torch.float32)

        return (img_t_minus_2, img_t_minus_1, img_t), state_seq_tensor, target_tensor

