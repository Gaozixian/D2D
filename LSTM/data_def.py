import ast
import numpy as np
import yaml
import csv

def load_config(config_path="./config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        # 解析 YAML 为字典
        config = yaml.safe_load(f)
    return config

# 提供了一种办法将csv读取的str转换为list
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
