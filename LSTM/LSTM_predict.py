import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import pandas as pd
import yaml
import csv
import matplotlib.pyplot as plt
from data_def import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# batch_first = true:代表输入输出的tensor是[batch_size, seq_len, input_seze]

"""
---------目前是正确添加了时间序列，以10为维度单位来做-------------
会将新的csv文件保存下来，然后读取这个csv文件，作为训练数据
"""

class LSTM_class(nn.Module):
    def __init__(self, input_size=10, hidden_size=50, num_layers=2, output_size=1, drop_out=0.3):
        super(LSTM_class, self).__init__()
        self.hidden_size = hidden_size  # 代表从每个隐藏层上的单元个数
        self.num_layers = num_layers    # 代表经过多少层才能得到最终输出
        self.input_size = input_size    # [batch_size, time_seq, dimension]中的dimension
        self.output_size = output_size  # 这里代表的输出维度，这个输出维度可以是未来的三个时刻也可以是……
        self.drop_out = drop_out
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_out)
        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(hidden_size, output_size)
        # 初始化
    def forward(self, x):   # x:(batch_size, time_seq, input_size)
        batch_size = x.size(0)
        out, _ = self.LSTM(x)
        out = self.dropout(out[:, -1, :])
        output = self.fc(out)    # 取出最后一个时间步的维度，然后放到fc中
        return output


class MyDataset(Dataset):
    def __init__(self, file_name, img_path=None, view_path=None, normalizer=None):
        self.data = pd.read_csv(file_name)[['time_frame',
                                            'speed_ago', 'speed_now',
                                            'yaw_ago', 'yaw_now',
                                            'accel_ago', 'accel_now']]
        self.data['speed_ago'] = self.data['speed_ago'].apply(str_or_float_to_list)
        self.data['speed_now'] = self.data['speed_now'].apply(str_or_float_to_list)
        self.data['yaw_ago'] = self.data['yaw_ago'].apply(str_or_float_to_list)
        self.data['yaw_now'] = self.data['yaw_now'].apply(str_or_float_to_list)
        self.data['accel_ago'] = self.data['accel_ago'].apply(str_or_float_to_list)
        self.data['accel_now'] = self.data['accel_now'].apply(str_or_float_to_list)
        print(type(self.data['speed_ago'][0]))
        print(type(self.data['speed_now'][0]))
        self.normalizer = normalizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        speed_ago = self.data['speed_ago'][index]
        speed_now = self.data['speed_now'][index]  # 预测当前帧速度 → shape (1,)
        yaw_ago = self.data['yaw_ago'][index]
        yaw_now = self.data['yaw_now'][index]
        accel_ago = self.data['accel_ago'][index]
        accel_now = self.data['accel_now'][index]
        if self.normalizer is not None:  # 使用归一化了
            speed_ago, yaw_ago, accel_ago = self.normalizer.normalize_input(speed_ago, yaw_ago, accel_ago)
            speed_now = self.normalizer.normalize_label(speed_now)
        input_ago = np.stack([speed_ago, yaw_ago, accel_ago], axis=1)  # (9, 3)
        input_ago = torch.tensor(input_ago, dtype=torch.float32)
        speed_now = torch.tensor(speed_now, dtype=torch.float32).unsqueeze(0)
        yaw_now = torch.tensor(yaw_now, dtype=torch.float32).unsqueeze(0)  # 增加维度 → (1,)
        accel_now = torch.tensor(accel_now, dtype=torch.float32).unsqueeze(0)
        # print("input_ago的class：", type(input_ago))  # input_ago的class： <class 'torch.Tensor'>
        # print("speed_now的class：", type(speed_now))  # speed_now的class： <class 'torch.Tensor'>
        # print("input_ago的shape：", input_ago.shape)  # input_ago的shape： torch.Size([9, 3])
        # print("yaw_now的shape：", yaw_now.shape)  # speed_now的shape： torch.Size([1, 1])
        return input_ago, speed_now



if __name__ == '__main__':
    config = load_config()
    original_data_path = config['original_data_path']
    data_path = config['data_path']
    control_data = pd.read_csv(original_data_path)
    print(control_data.shape)
    add = add_data_dimension(10, control_data)
    add.add_data_dimension()
    dataset = MyDataset(data_path)
    print(dataset.data.dtypes)
    """
    ---------到这里是数据集建立完毕--------------
    """

    num_layers = config['num_layers']          # LSTM 层数
    output_size = config['output_size']         # 最终输出（预测的转向角）
    seq_length = config['seq_length']         # 时序窗口长度（前9帧预测第10帧）
    input_size = config['input_size']           # 每个时间步的特征数：速度、偏航角、加速度
    hidden_size = config['hidden_size']         # LSTM隐藏单元数
    batch_size = config['batch_size']          # 批次大小
    epochs = config['epochs']             # 最大训练轮次
    lr = config['lr']               # 学习率
    val_ratio = config['val_ratio']        # 验证集比例（20%数据用于验证）
    patience = config['patience']             # 早停耐心值（5轮验证集loss不下降则停止）
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stop_count = 0

    # --------------开始构建数据集----------------------
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    print(f"训练集样本数：{len(train_dataset)}，验证集样本数：{len(val_dataset)}")
    print(f"训练集批次：{len(train_dataloader)}，验证集批次：{len(val_dataloader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_class(input_size, hidden_size, num_layers)
    model.to(device)
    loss_fun = nn.MSELoss()  # 回归任务使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # weight_decay是L2正则化，防止过拟合
    Loss = 10
    for epoch in range(epochs):
        print(f"第{epoch}轮训练开始")
        model.train()
        train_loss = 0.0
        for i, data in enumerate(train_dataloader):
            input_truth, output_truth = data
            # print("input_truth的class：", type(input_truth))
            input_truth = input_truth.to(device)    # input_truth的class： <class 'torch.Tensor'>
            output_truth = output_truth.to(device)
            optimizer.zero_grad()
            out_pre = model(input_truth)
            loss = loss_fun(out_pre*100, output_truth*100)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            train_loss += loss.item() * input_truth.size(0)
        avg_train_loss = train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        # 新增：每轮验证时临时存储该轮的预测和真实值
        epoch_preds = []
        epoch_truths = []

        with torch.no_grad():
            for X, y in val_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fun(pred, y)
                val_loss += loss.item() * X.size(0)

                # 收集预测值和真实值（转移到CPU并转为numpy）
                epoch_preds.extend(pred.cpu().numpy().flatten())
                epoch_truths.extend(y.cpu().numpy().flatten())

        avg_val_loss = val_loss / len(val_dataset)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch + 1:2d}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f}")

        # 保存最佳模型时，同时记录该轮的预测结果
        all_preds = epoch_preds
        all_truths = epoch_truths
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_lstm_model.pth')
            # 保存当前最佳的预测结果和真实值
            all_preds = epoch_preds
            all_truths = epoch_truths
            print(f"  → 验证损失下降，保存最佳模型和预测结果（当前最佳Loss: {best_val_loss:.6f}）")
            early_stop_count = 0
        else:
            early_stop_count += 1
            print(f"  → 验证损失未下降，早停计数: {early_stop_count}/{patience}")
            if early_stop_count >= patience:
                print(f"\n早停触发！最佳验证Loss: {best_val_loss:.6f}")
                break

    # 训练结束后保存预测结果到CSV文件
    results = pd.DataFrame({
        'Truth': all_truths,
        'Preds': all_preds
    })
    results.to_csv('speed_prediction_results.csv', index=False, encoding='utf-8')
    print("预测结果已保存到 'speed_prediction_results.csv'")

    # 绘制预测 vs 真实速度曲线
    if True:
        plt.figure(figsize=(12, 6))

        plt.plot(range(len(all_truths)), all_truths, label='真实速度', alpha=0.7, linewidth=2, color='blue')
        plt.plot(range(len(all_preds)), all_preds, label='预测速度', alpha=0.7, linestyle='--', linewidth=2, color='red')
        plt.xlabel('样本索引')
        plt.ylabel('速度值')
        plt.title('LSTM模型速度预测 vs 真实值对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # plt.savefig('speed_prediction_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig('steer_prediction_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    # ---------------------- 训练结束后可视化损失曲线 ----------------------
    if True:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='训练损失', linewidth=2)
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='验证损失', linewidth=2)
        plt.xlabel('训练轮次 (Epoch)')
        plt.ylabel('均方误差 (MSE)')
        plt.title('LSTM模型训练损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('steer_loss_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\n训练完成！最佳模型已保存为 'best_lstm_model.pth'")