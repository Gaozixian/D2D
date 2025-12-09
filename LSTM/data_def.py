import ast
import numpy as np
import yaml
import csv

# 提供了一种办法将csv读取的str转换为list
# def str_to_list(s):
#     try:
#         # 用 ast.literal_eval 解析字符串列表
#         return ast.literal_eval(s)
#     except (ValueError, SyntaxError):
#         # 若解析失败，返回空列表或原数据（根据需求调整）
#         return []
def load_config(config_path="./config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        # 解析 YAML 为字典
        config = yaml.safe_load(f)
    return config

# 2. 加载配置

def str_or_float_to_list(s):
    try:
        # 如果是字符串，用 ast.literal_eval 解析
        if isinstance(s, str):
            return ast.literal_eval(s)
        # 如果是数字，转换为单元素列表
        elif isinstance(s, (int, float, np.float64)):
            return [float(s)]
        # 如果已经是列表，直接返回
        elif isinstance(s, list):
            return s
        else:
            return []
    except (ValueError, SyntaxError):
        return []

def plt_pred_truth(csv_path):
    # 读取数据
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

class Normalizer:   # 归一化类
    def __init__(self):
        self.speed_mean = 0.0
        self.speed_std = 1.0
        self.yaw_mean = 0.0
        self.yaw_std = 1.0
        self.accel_mean = 0.0
        self.accel_std = 1.0

    def fit(self, dataset):
        """根据训练数据计算均值和标准差"""
        all_speed_ago = []
        all_yaw_ago = []
        all_accel_ago = []
        all_speed_now = []

        # 收集所有数据
        for i in range(len(dataset)):
            speed_ago = dataset.data['speed_ago'][i]
            yaw_ago = dataset.data['yaw_ago'][i]
            accel_ago = dataset.data['accel_ago'][i]
            speed_now = dataset.data['speed_now'][i]
            
            all_speed_ago.extend(speed_ago)
            all_yaw_ago.extend(yaw_ago)
            all_accel_ago.extend(accel_ago)
            all_speed_now.extend(speed_now)

        # 计算均值和标准差（避免除零）
        self.speed_mean = np.mean(all_speed_ago) if all_speed_ago else 0.0
        self.speed_std = np.std(all_speed_ago) if np.std(all_speed_ago) > 1e-6 else 1.0
        
        self.yaw_mean = np.mean(all_yaw_ago) if all_yaw_ago else 0.0
        self.yaw_std = np.std(all_yaw_ago) if np.std(all_yaw_ago) > 1e-6 else 1.0
        
        self.accel_mean = np.mean(all_accel_ago) if all_accel_ago else 0.0
        self.accel_std = np.std(all_accel_ago) if np.std(all_accel_ago) > 1e-6 else 1.0

    def normalize_input(self, speed_ago, yaw_ago, accel_ago):
        """归一化输入特征（前序序列）"""
        speed_norm = (np.array(speed_ago) - self.speed_mean) / self.speed_std
        yaw_norm = (np.array(yaw_ago) - self.yaw_mean) / self.yaw_std
        accel_norm = (np.array(accel_ago) - self.accel_mean) / self.accel_std
        return speed_norm, yaw_norm, accel_norm

    def normalize_label(self, speed_now):
        """归一化标签（当前速度）"""
        return (np.array(speed_now) - self.speed_mean) / self.speed_std

    def denormalize(self, speed_norm):
        """反归一化预测结果（恢复真实尺度）"""
        return np.array(speed_norm) * self.speed_std + self.speed_mean

class add_data_dimension:
    def __init__(self, seq_len, data):
        self.data = data
        self.seq_len = seq_len
    def add_data_dimension(self):
        csv_file = 'sample_data.csv'
        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # 写入表头
            headers = ['time_frame', 'speed_now', 'yaw_now', 'accel_now', 'speed_ago', 'yaw_ago', 'accel_ago']
            writer.writerow(headers)
            # 开始处理增加前序的维度
            for index in range(len(self.data)): # index代表当前在第几行
                yaws = []   # clear list for each now
                speeds = []
                accels = []
                for i in range(self.seq_len):
                    add_index = max(0, index - i)
                    sp = self.data.at[add_index, 'speed_kmh']
                    ya = self.data.at[add_index, 'steer']
                    acc = self.data.at[add_index, 'throttle']
                    speeds.append(sp)
                    yaws.append(ya)
                    accels.append(acc)
                speeds.reverse()
                yaws.reverse()
                accels.reverse()
                time_frame = self.data.at[index, 'timestamp']
                speed_now = self.data.at[index, 'speed_kmh']
                yaw_now = self.data.at[index, 'steer']
                accel_now = self.data.at[index, 'throttle']
                speed_ago = speeds[:-1]
                yaw_ago = yaws[:-1]
                accel_ago = accels[:-1]
                data = [time_frame, speed_now, yaw_now, accel_now, speed_ago, yaw_ago, accel_ago]
                writer.writerow(data)
                print(f'已写入第{index}行数据')
