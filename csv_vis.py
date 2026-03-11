import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False


def generate_asymmetric_driving_data(num_samples=60000, seed=42):
    """
    生成非对称分布的自动驾驶数据
    """
    np.random.seed(seed)

    # 1. 生成全局速度分布 (Beta分布模拟城市+高速混合)
    speed = np.random.beta(2.0, 3.0, num_samples)

    # 2. 定义非对称锚点
    # 速度点
    speed_anchors = np.array([0.0, 0.2, 0.5, 0.8, 1.0])

    # 转角标准差锚点 (控制漏斗形状)
    steer_std_anchors = np.array([0.5, 0.35, 0.2, 0.08, 0.02])

    # 转角均值锚点 (控制中心偏移，打破对称)
    # 模拟低速时路口左转动作更多或幅度更大的情况
    steer_mean_anchors = np.array([0.08, 0.04, 0.01, 0.0, 0.0])

    # 3. 插值计算每个样本的参数
    current_stds = np.interp(speed, speed_anchors, steer_std_anchors)
    current_means = np.interp(speed, speed_anchors, steer_mean_anchors)

    # 4. 生成非对称转角
    # 技巧：使用 Gumbel 分布或偏正态的思想，或者简单的扰动
    # 这里通过在正态分布基础上增加一个与速度相关的非对称随机项
    steering = np.random.normal(current_means, current_stds)

    # 额外添加：低速下的非对称长尾 (例如更多的左转样本)
    urban_mask = speed < 0.3
    num_urban = np.sum(urban_mask)
    # 给低速区增加一个向左的偏差
    steering[urban_mask] += np.random.uniform(-0.15, 0.05, num_urban)

    # 5. 裁剪范围
    speed = np.clip(speed, 0, 1)
    steering = np.clip(steering, -1, 1)

    return speed, steering


def plot_asymmetric_heatmaps(speed, steering):
    fig, ax = plt.subplots(figsize=(12, 9))

    # 使用 hexbin 绘制热力图
    # cmap 使用 'magma' 或 'viridis' 能更好地观察非对称边缘
    hb = ax.hexbin(speed, steering, gridsize=70, cmap='magma',
                   bins='log', mincnt=1, extent=[0, 1, -1, 1])

    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('样本密度 (Log10)', fontsize=12)

    ax.set_title('非对称自动驾驶数据分布热力图\n(模拟真实道路偏置)', fontsize=15, pad=15)
    ax.set_xlabel('归一化速度 (v)', fontsize=12)
    ax.set_ylabel('归一化转角 (δ)', fontsize=12)

    # 辅助线：0刻度线，方便对比非对称性
    ax.axhline(0, color='cyan', linestyle=':', alpha=0.5, label='绝对零点')

    # 绘制数据重心线
    s_line = np.linspace(0, 1, 50)
    # m_line = np.interp(s_line, [0.0, 0.2, 0.5, 0.8, 1.0], [0.08, 0.04, 0.01, 0.0, 0.0])
    # ax.plot(s_line, m_line, 'w--', alpha=0.7, label='分布重心偏置')

    ax.grid(True, alpha=0.1)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    s, d = generate_asymmetric_driving_data()
    plot_asymmetric_heatmaps(s, d)



## 自定义速度区间生成
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False


def generate_custom_driving_data(scenarios, seed=42):
    """
    根据配置的场景列表生成合成数据

    参数 scenarios 格式:
    [
        {"name": "高速", "speed_range": (0.7, 1.0), "count": 10000, "steer_std": 0.05, "steer_mean": 0.0},
        ...
    ]
    """
    np.random.seed(seed)
    all_speeds = []
    all_steers = []

    print(f"{'场景名称':<10} | {'样本量':<8} | {'速度区间':<12} | {'转角标准差'}")
    print("-" * 50)

    for sc in scenarios:
        count = sc['count']
        s_min, s_max = sc['speed_range']

        # 1. 在区间内均匀或正态生成速度
        # 这里使用均匀分布模拟区间覆盖，也可以改用 np.random.normal
        speeds = np.random.uniform(s_min, s_max, count)

        # 2. 根据该速度区间的设定生成转角分布
        # steer_mean 通常为0（直行为主），steer_std 决定了转弯的幅度
        steers = np.random.normal(sc.get('steer_mean', 0.0), sc['steer_std'], count)

        all_speeds.append(speeds)
        all_steers.append(steers)

        print(f"{sc['name']:<10} | {count:<10} | {s_min:>4.1f}-{s_max:<4.1f} | {sc['steer_std']:.3f}")

    # 合并并裁剪
    speed_array = np.concatenate(all_speeds)
    steer_array = np.concatenate(all_steers)

    return np.clip(speed_array, 0, 1), np.clip(steer_array, -1, 1)


def plot_heatmaps(speed, steering):
    """绘制热力图可视化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 1. Hexbin 热力图 (适合观察聚类)
    hb = ax1.hexbin(speed, steering, gridsize=50, cmap='magma', mincnt=1, extent=[0, 1, -1, 1])
    fig.colorbar(hb, ax=ax1, label='样本密度')
    ax1.set_title('自动驾驶数据分布 (Hexbin)', fontsize=15)
    ax1.set_xlabel('速度 (Normalized)')
    ax1.set_ylabel('转角 (Normalized)')

    # 2. 2D 直方图 (适合观察边界)
    h2d = ax2.hist2d(speed, steering, bins=60, cmap='viridis', range=[[0, 1], [-1, 1]])
    fig.colorbar(h2d[3], ax=ax2, label='计数')
    ax2.set_title('速度-转角 2D 直方图', fontsize=15)
    ax2.set_xlabel('速度')

    plt.tight_layout()
    plt.show()


def main():
    # --- 核心配置区：在这里自由设定你的数据区间 ---
    # steer_std 越大，代表该速度下转弯动作越剧烈/频繁
    config = [
        {
            "name": "高速巡航",
            "speed_range": (0.7, 1.0),
            "count": 15000,
            "steer_std": 0.02,  # 高速时转角非常微小
            "steer_mean": 0.0
        },
        {
            "name": "城市主干道",
            "speed_range": (0.3, 0.7),
            "count": 20000,
            "steer_std": 0.15,  # 中速有一定幅度的转向
            "steer_mean": 0.0
        },
        {
            "name": "低速弯道/路口",
            "speed_range": (0.1, 0.3),
            "count": 10000,
            "steer_std": 0.4,  # 低速时允许大角度转弯
            "steer_mean": 0.05
        },
        {
            "name": "极端泊车",
            "speed_range": (0.0, 0.1),
            "count": 5000,
            "steer_std": 0.8,  # 极低速时转角分布极广
            "steer_mean": 0.0
        }
    ]

    print("开始生成自定义分布数据...")
    speed, steer = generate_custom_driving_data(config)

    print(f"\n总计样本量: {len(speed)}")
    plot_heatmaps(speed, steer)


if __name__ == "__main__":
    main()