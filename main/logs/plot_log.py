import re
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os


def parse_log_file(file_path):
    """
    解析我们自定义的日志文件格式
    """
    batch_data = []
    epoch_data = []

    # 匹配 Batch 级别的日志
    batch_pattern = re.compile(
        r"Epoch \[(\d+)/\d+\], Batch \[(\d+)/\d+\] \| "
        r"真实误差 -> Vel: ([\d\.]+), Steer: ([\d\.]+) \| "
        r"动态权重 -> Vel_W: ([\d\.]+), Steer_W: ([\d\.]+) \| "
        r"\(总优化目标: ([\-\d\.]+)\)"
    )

    # 匹配 Epoch 级别的日志总结
    epoch_pattern = re.compile(
        r"Epoch \[(\d+)/\d+\] \| "
        r"Total Loss: ([\-\d\.]+) \| "
        r"Vel Loss: ([\d\.]+) \| "
        r"Steer Loss: ([\d\.]+) \| "
        r"Vel_W: ([\d\.]+) \| "
        r"Steer_W: ([\d\.]+)"
    )

    if not os.path.exists(file_path):
        print(f"找不到文件: {file_path}")
        return None, None

    with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            line = line.strip()

            # 尝试匹配 Batch 级日志
            b_match = batch_pattern.search(line)
            if b_match:
                batch_data.append({
                    'Epoch': int(b_match.group(1)),
                    'Batch': int(b_match.group(2)),
                    'Vel_Loss': float(b_match.group(3)),
                    'Steer_Loss': float(b_match.group(4)),
                    'Vel_W': float(b_match.group(5)),
                    'Steer_W': float(b_match.group(6)),
                    'Total_Loss': float(b_match.group(7))
                })
                continue

            # 尝试匹配 Epoch 级日志
            e_match = epoch_pattern.search(line)
            if e_match:
                epoch_data.append({
                    'Epoch': int(e_match.group(1)),
                    'Total_Loss': float(e_match.group(2)),
                    'Vel_Loss': float(e_match.group(3)),
                    'Steer_Loss': float(e_match.group(4)),
                    'Vel_W': float(e_match.group(5)),
                    'Steer_W': float(e_match.group(6))
                })

    df_batch = pd.DataFrame(batch_data)
    df_epoch = pd.DataFrame(epoch_data)
    return df_batch, df_epoch


def plot_training_curves(df_batch, df_epoch, save_name="training_curves.png"):
    """
    绘制并保存训练曲线图
    """
    if df_batch.empty:
        print("未解析到 Batch 数据，请检查日志格式是否匹配！")
        return

    # 创建一个 2x2 的子图网格
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('End-to-End Driving Model Training Logs', fontsize=18)

    # 全局 Batch 索引 (用于 X 轴)
    global_batch_steps = range(len(df_batch))

    # --- 图 1: Batch 级别的真实物理误差 (Huber Loss) ---
    axs[0, 0].plot(global_batch_steps, df_batch['Vel_Loss'], label='Velocity Error (Huber)', alpha=0.7, color='blue')
    axs[0, 0].plot(global_batch_steps, df_batch['Steer_Loss'], label='Steer Error (Huber)', alpha=0.7, color='orange')
    axs[0, 0].set_title('Real Physical Error (per Batch)', fontsize=14)
    axs[0, 0].set_xlabel('Total Batches')
    axs[0, 0].set_ylabel('Huber Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)

    # --- 图 2: Batch 级别的动态权重变化 (Uncertainty Weights) ---
    axs[0, 1].plot(global_batch_steps, df_batch['Vel_W'], label='Velocity Weight (Vel_W)', linewidth=2, color='blue')
    axs[0, 1].plot(global_batch_steps, df_batch['Steer_W'], label='Steer Weight (Steer_W)', linewidth=2, color='orange')
    axs[0, 1].set_title('Dynamic Task Weights Adaptation', fontsize=14)
    axs[0, 1].set_xlabel('Total Batches')
    axs[0, 1].set_ylabel('Weight Value exp(-s)')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)

    # --- 图 3: Batch 级别的总优化目标 (带负数的那个) ---
    axs[1, 0].plot(global_batch_steps, df_batch['Total_Loss'], label='Total AWL Loss', color='purple', alpha=0.8)
    axs[1, 0].set_title('Optimizer Total Target (Automatic Weighted Loss)', fontsize=14)
    axs[1, 0].set_xlabel('Total Batches')
    axs[1, 0].set_ylabel('Loss Value (Can be negative)')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)

    # --- 图 4: Epoch 级别的真实误差均值总结 ---
    if not df_epoch.empty:
        epochs = df_epoch['Epoch']
        axs[1, 1].plot(epochs, df_epoch['Vel_Loss'], label='Avg Velocity Error', marker='o', linewidth=2)
        axs[1, 1].plot(epochs, df_epoch['Steer_Loss'], label='Avg Steer Error', marker='s', linewidth=2)
        axs[1, 1].set_title('Average Error (per Epoch)', fontsize=14)
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Huber Loss')
        # 强制 x 轴显示整数
        from matplotlib.ticker import MaxNLocator
        axs[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[1, 1].legend()
        axs[1, 1].grid(True, linestyle='--', alpha=0.6)
    else:
        axs[1, 1].text(0.5, 0.5, 'No Epoch data found yet (Train longer)',
                       horizontalalignment='center', verticalalignment='center')
        axs[1, 1].set_title('Average Error (per Epoch)')

    # 调整布局并保存/显示
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_name, dpi=300)
    print(f"图表已成功保存为: {save_name}")
    plt.show()


if __name__ == "__main__":
    # 使用 argparse 处理命令行参数
    parser = argparse.ArgumentParser(description="Visualize Training Logs for Driving Model")
    parser.add_argument("log_file", type=str, help="日志文件的路径 (例如: logs/training_log_20231024.txt)")
    parser.add_argument("--save_name", type=str, default="training_curves.png", help="保存的图片名称")

    args = parser.parse_args()

    # 解析并画图
    print(f"正在解析日志: {args.log_file} ...")
    df_batch, df_epoch = parse_log_file(args.log_file)

    if df_batch is not None:
        plot_training_curves(df_batch, df_epoch, save_name=args.save_name)