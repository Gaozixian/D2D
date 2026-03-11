import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False


def generate_merged_driving_data(scenarios, seed=42):
    """
    功能增强：正速度平滑过渡，负速度严格按设定生成（不平滑）
    """
    np.random.seed(seed)

    # 分类存储数据
    pos_anchors_speed = []
    pos_anchors_std = []
    pos_anchors_mean = []

    final_speeds = []
    final_steers = []

    print(f"{'场景':<10} | {'速度区间':<10} | {'状态':<8} | {'标准差'}")
    print("-" * 60)

    for sc in scenarios:
        s_min, s_max = sc['speed_range']
        count = sc['count']
        dist_type = sc.get('speed_dist', 'uniform')

        # 1. 生成该区间内的速度
        if dist_type == 'uniform':
            v = np.random.uniform(s_min, s_max, count)
        elif dist_type == 'normal':
            mu = (s_min + s_max) / 2
            sigma = (s_max - s_min) / 6
            v = np.random.normal(mu, sigma, count)
            v = np.clip(v, s_min, s_max)
        else:
            v = np.random.uniform(s_min, s_max, count)

        # 2. 逻辑分支处理
        if s_max <= 0:
            # --- 负速度逻辑：严格匹配设定，不参与插值 ---
            # 直接使用配置中的 mean 和 std 生成
            s = np.random.normal(sc.get('steer_mean', 0.0), sc['steer_std'], count)
            final_speeds.append(v)
            final_steers.append(s)
            print(f"{sc['name']:<10} | {s_min:>4.1f}-{s_max:<4.1f} | 严格固定 | {sc['steer_std']:.3f}")
        else:
            # --- 正速度逻辑：收集锚点准备插值 ---
            center_speed = (s_min + s_max) / 2
            pos_anchors_speed.append(center_speed)
            pos_anchors_std.append(sc['steer_std'])
            pos_anchors_mean.append(sc.get('steer_mean', 0.0))

            # 先暂存速度，稍后统一插值计算转角
            # 我们用一个临时标记来记录这部分数据需要平滑处理
            final_speeds.append(v)
            final_steers.append(None)  # 占位符
            print(f"{sc['name']:<10} | {s_min:>4.1f}-{s_max:<4.1f} | 平滑过渡 | {sc['steer_std']:.3f}")

    # 3. 处理正速度区域的平滑插值
    # 提取所有需要平滑的速度点
    all_v = np.concatenate(final_speeds)

    # 准备正速度锚点排序
    sort_idx = np.argsort(pos_anchors_speed)
    as_arr = np.array(pos_anchors_speed)[sort_idx]
    astd_arr = np.array(pos_anchors_std)[sort_idx]
    am_arr = np.array(pos_anchors_mean)[sort_idx]

    # 重新构建最终结果
    actual_final_steers = []
    idx_offset = 0
    for i, sc in enumerate(scenarios):
        v_part = final_speeds[i]
        if final_steers[i] is not None:
            # 负速度部分：直接添加已生成的
            actual_final_steers.append(final_steers[i])
        else:
            # 正速度部分：计算插值
            current_stds = np.interp(v_part, as_arr, astd_arr)
            current_means = np.interp(v_part, as_arr, am_arr)
            s_part = np.random.normal(current_means, current_stds)

            # 仅对正速度区域保留原有的局部扰动（可选）
            urban_mask = (v_part > 0) & (v_part < 0.3)
            if np.any(urban_mask):
                s_part[urban_mask] += np.random.uniform(-0.1, 0.1, np.sum(urban_mask))

            actual_final_steers.append(s_part)

    speed_array = np.concatenate(final_speeds)
    steer_array = np.concatenate(actual_final_steers)

    return np.clip(speed_array, -1, 1), np.clip(steer_array, -1, 1)


def plot_merged_visualization(speed, steering, output_path='speed_steering_heatmap.png'):
    """
    只绘制并保存左侧的 Hexbin 热力图
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    # 设定一个显示倍数
    display_multiplier = 15.0
    # 为每个点分配一个权重
    weights = np.ones_like(speed) * display_multiplier

    # 使用 hexbin 绘制，extent 范围扩大到包含负数速度
    hb = ax.hexbin(speed, steering,
                   C=weights,
                   reduce_C_function=np.sum,  # 将权重求和作为显示数值
                   gridsize=70,
                   cmap='magma',
                   bins='log',
                   mincnt=1,
                   extent=[-0.15, 1, -1, 1])

    fig.colorbar(hb, ax=ax, label='样本密度')

    ax.set_title('样本分布热力图', fontsize=15)
    ax.set_xlabel('速度 (v)', fontsize=15)
    ax.set_ylabel('转角 (δ)', fontsize=15)

    # 辅助线
    ax.axhline(0, color='white', linestyle=':', alpha=0.3)
    ax.axvline(0, color='cyan', linestyle='--', alpha=0.4, label='零速线')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"热力图已成功保存至: {output_path}")
    plt.show()
    plt.close()


def main():
    config = [
        {
            "name": "倒车/负向",
            "speed_range": (-0.1, 0.05),
            "count": 500,
            "speed_dist": "uniform",
            "steer_std": 0,  # 负速度下会严格保持这个宽度，不被正速度干扰
            "steer_mean": 0.0
        },
        {
            "name": "极低速",
            "speed_range": (0.0, 0.4),
            "count": 2000,
            "speed_dist": "normal",
            "steer_std": 0.04,
            "steer_mean": 0.01
        },
        {
            "name": "城市/弯道",
            "speed_range": (0.1, 0.6),
            "count": 1000,
            "steer_std": 0.35,
            "steer_mean": 0.01
        },
        {
            "name": "中速弯道",
            "speed_range": (0.25, 0.6),
            "count": 4000,
            "steer_std": 0.35,
            "steer_mean": 0.01
        },
        {
            "name": "高速巡航",
            "speed_range": (0.4, 0.7),
            "count": 5000,
            "steer_std": 0.25,
            "steer_mean": -0.01
        },
        {
            "name": "超高速直行",
            "speed_range": (0.5, 1.0),
            "count": 4000,
            "steer_std": 0.02,
            "steer_mean": 0.0
        }
    ]

    print("开始生成混合逻辑数据...")
    speed, steer = generate_merged_driving_data(config)
    plot_merged_visualization(speed, steer)


if __name__ == "__main__":
    main()