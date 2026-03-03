import re
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import signal

# 设置中文字体（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def parse_loss_log(file_path):
    """解析log文件，提取epoch、batch、loss数据"""
    loss_data = []
    epoch_list = []
    batch_list = []

    pattern = r'Epoch \[(\d+)/\d+\], Batch \[(\d+)/\d+\], Loss: ([\d\.]+)'

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line.strip())
            if match:
                epoch = int(match.group(1))
                batch = int(match.group(2))
                loss = float(match.group(3))

                loss_data.append(loss)
                epoch_list.append(epoch)
                batch_list.append(batch)

    return epoch_list, batch_list, loss_data


def plot_loss_curve_sampled(file_paths,
                            smooth_window=11,
                            y_lim=None,
                            highlight_epochs=True,
                            start_epoch=1,
                            sample_step=10):  # <--- 新增参数：采样步长，默认隔10个取1个
    """
    采样版Loss绘图：隔10个Batch取1个数据点绘图
    1. 隔10次打印一次Loss，减少数据量
    2. 保持Epoch顺序和紧凑布局
    """
    plt.figure(figsize=(16, 9))  # 紧凑画布比例
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    all_sampled_loss = []  # 收集采样后的loss，用于纵轴适配

    for idx, file_path in enumerate(file_paths):
        if not os.path.exists(file_path):
            print(f"文件 {file_path} 不存在，跳过")
            continue

        # 1. 解析完整数据
        epoch_list, batch_list, loss_data = parse_loss_log(file_path)
        if not loss_data:
            print(f"文件 {file_path} 中未提取到Loss数据")
            continue

        # 2. 先截断起始Epoch前的数据
        try:
            start_idx = epoch_list.index(start_epoch)
        except ValueError:
            print(f"文件 {file_path} 中未找到 Epoch {start_epoch}，跳过绘图")
            continue

        truncated_epochs = epoch_list[start_idx:]
        truncated_loss = loss_data[start_idx:]
        truncated_batch = list(range(start_idx + 1, len(loss_data) + 1))  # 截断后的全局batch

        # 3. 核心：隔10个Batch采样一次数据（步长采样）
        # 采样规则：取第0、10、20...个数据点（索引从0开始）
        sampled_indices = list(range(0, len(truncated_loss), sample_step))
        sampled_epochs = [truncated_epochs[i] for i in sampled_indices]
        sampled_loss = [truncated_loss[i] for i in sampled_indices]
        sampled_batch = [truncated_batch[i] for i in sampled_indices]

        all_sampled_loss.extend(sampled_loss)  # 收集采样后的数据

        # 4. 可选：对采样后的数据做平滑（可选，采样后波动已减少）
        if smooth_window > 1 and len(sampled_loss) > smooth_window:
            win_len = min(smooth_window, len(sampled_loss))
            if win_len % 2 == 0: win_len -= 1
            loss_smoothed = signal.savgol_filter(sampled_loss, win_len, 2)
        else:
            loss_smoothed = sampled_loss

        # 5. 绘制采样后的曲线
        file_name = os.path.basename(file_path)
        # 采样后的原始点（用散点+折线，更清晰）
        # plt.plot(sampled_batch, sampled_loss, color=colors[idx % len(colors)],
        #          alpha=0.5, linewidth=1, marker='', markersize=3, label=f'{file_name}（采样原始）')
        # 采样后的平滑曲线（突出趋势）
        plt.plot(sampled_batch, loss_smoothed, color=colors[idx % len(colors)],
                 alpha=0.9, linewidth=1.5, label=f'{file_name}（采样平滑）')

    # 6. 绘制Epoch分隔线（适配采样后的数据）
    if highlight_epochs and all_sampled_loss:
        first_fp = file_paths[0]
        te_full, _, _ = parse_loss_log(first_fp)
        try:
            start_idx = te_full.index(start_epoch)
        except ValueError:
            start_idx = 0

        # 找到采样后的Epoch分界点
        te_truncated = te_full[start_idx:]
        sampled_indices = list(range(0, len(te_truncated), sample_step))
        te_sampled = [te_truncated[i] for i in sampled_indices]
        tb_truncated = list(range(start_idx + 1, len(te_full) + 1))
        tb_sampled = [tb_truncated[i] for i in sampled_indices]

        if te_sampled:
            prev_epoch = te_sampled[0]
            for i, epoch in enumerate(te_sampled):
                if epoch > prev_epoch:
                    batch_pos = tb_sampled[i]
                    plt.axvline(x=batch_pos, color='gray', linestyle='--', alpha=0.7, linewidth=1)
                    # 标注下移，避免纵向拉长
                    plt.text(batch_pos + 5, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.1,
                             f'Epoch {epoch}', fontsize=8, color='black')
                    prev_epoch = epoch

    # 7. 优化纵轴范围（基于采样后的数据）
    if y_lim:
        y_min = y_lim[0] if y_lim[0] is not None else np.min(all_sampled_loss) if all_sampled_loss else 0
        y_max = y_lim[1] if y_lim[1] is not None else np.max(all_sampled_loss) if all_sampled_loss else 1
        plt.ylim(y_min, y_max)
    else:
        if all_sampled_loss:
            q2 = np.percentile(all_sampled_loss, 2)
            q98 = np.percentile(all_sampled_loss, 98)
            plt.ylim(q2, q98)

    # 图表样式
    plt.xlabel(f'全局Batch序号（从 Epoch {start_epoch} 开始，隔{sample_step}个采样1个）', fontsize=12)
    plt.ylabel('Loss值', fontsize=12)
    plt.title(f'Loss变化曲线（隔{sample_step}个Batch采样）', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1, 0.95))

    plt.tight_layout()
    plt.savefig(f'loss_curve_sampled_step{sample_step}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()


# ------------------- 调用示例 -------------------
log_file_paths = [
    # 'a=0.3.txt',
    # 'a=0.5.txt',
    'v_a0.5.txt'
]

# 调用：隔10个Batch采样一次（核心参数sample_step=10）
plot_loss_curve_sampled(
    file_paths=log_file_paths,
    smooth_window=15,  # 采样后波动减少，可减小平滑窗口
    start_epoch=1,  # 从第2个Epoch开始
    sample_step=1,  # 隔10个取1个（关键参数）
    y_lim=[0, 0.05]  # 手动指定y轴范围
)