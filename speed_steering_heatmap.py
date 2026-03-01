#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动驾驶数据速度与转角热力图可视化
=====================================
功能：
    1. 生成符合自动驾驶场景分布的合成数据
    2. 绘制速度和转角的二维热力图
    3. 数据密集区域颜色越深

作者：MiniMax Agent
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
import os

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False


def generate_autonomous_driving_data(num_samples=19000, seed=41):
    """
    生成符合自动驾驶数据集分布的合成数据

    数据生成逻辑：
    1. 速度使用Beta分布，偏向低速（城市驾驶场景）
    2. 转角使用正态分布，中心为0（直线行驶为主）
    3. 速度与转角呈负相关：高速时转角小，低速时转角大（物理约束）

    参数:
        num_samples: 数据点数量
        seed: 随机种子，确保可复现

    返回:
        speed: 归一化速度数组 (0-1)
        steering: 归一化转角数组 (-1 到 1)
    """
    np.random.seed(seed)

    # 步骤1：生成速度数据
    # 使用Beta分布，参数alpha>beta使数据偏向低速
    # alpha=2, beta=5 产生一个左偏的分布，大部分在0-0.4区间
    speed = np.random.beta(2.5, 4, num_samples)

    # 步骤2：生成转角数据（考虑速度的影响）
    # 高速时转角应该较小，低速时转角可以较大
    # 使用与速度相关的标准差来实现这个物理约束
    steering = np.zeros(num_samples)

    for i in range(num_samples):
        # 速度越低，转角范围越大；速度越高，转角范围越小
        # 基础标准差为0.3，随着速度增加而减小
        scale = 0.3 * (1 - speed[i] * 0.8)  # 确保高速时转角接近0
        steering[i] = np.random.normal(0.1, scale)

    # 步骤3：添加一些典型的驾驶场景
    # 3.1 高速巡航场景（速度>0.7，转角接近0）- 增加这部分数据
    num_highway = int(num_samples * 0.15)
    highway_speeds = np.random.beta(6, 1.5, num_highway)  # 更集中在高速区间(0.7-1.0)
    highway_steering = np.random.normal(0, 0.03, num_highway)  # 极小的转角，更集中在0附近

    # 3.2 城市中低速转向场景（速度0.2-0.5，转角较大）- 新增
    num_turning = int(num_samples * 0.5)  # 增加15%的中低速转向数据
    turning_speeds = np.random.beta(4, 4, num_turning)  # 集中在0.3-0.6区间
    # 生成明显的左转或右转（不包含0附近的小转角）
    turning_directions = np.random.choice([-1, 1], num_turning)  # 随机左/右转
    turning_steering = turning_directions * (0.3 + np.random.beta(2, 1.5, num_turning) * 0.5)

    # 3.3 城市低速转弯场景（速度<0.3，转角较大）
    num_urban = int(num_samples * 0.25)
    urban_speeds = np.random.beta(3, 3, num_urban)  # 偏向低速
    urban_steering = np.random.normal(0.01, 0.25, num_urban)  # 较大的转角

    # 3.4 停车场/掉头场景（速度很低，转角很大）
    num_parking = int(num_samples * 0.03)
    parking_speeds = np.random.beta(1, 6, num_parking)  # 非常低的速度
    parking_steering = np.random.normal(0, 0.4, num_parking)  # 大转角

    # 合并所有数据
    speed = np.concatenate([speed, highway_speeds, turning_speeds, urban_speeds, parking_speeds])
    steering = np.concatenate([steering, highway_steering, turning_steering, urban_steering, parking_steering])

    # 步骤4：裁剪到合理范围
    speed = np.clip(speed, 0, 1)
    steering = np.clip(steering, -1, 1)

    # 打乱数据顺序
    indices = np.random.permutation(len(speed))
    speed = speed[indices]
    steering = steering[indices]

    return speed, steering


def create_heatmap(speed, steering, output_path='speed_steering_heatmap.png'):
    """
    创建速度与转角的热力图

    参数:
        speed: 归一化速度数组
        steering: 归一化转角数组
        output_path: 输出图片路径
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 9))

    # 使用hexbin绘制六边形热力图（适合密集数据）
    # gridsize控制六边形大小
    hb = ax.hexbin(
        speed,
        steering,
        gridsize=50,  # 六边形网格大小
        cmap='inferno',  # 热力图配色：低密度浅色，高密度深色
        mincnt=1,  # 至少1个点才显示
        extent=[0, 1, -1, 1]  # 坐标范围
    )

    # 添加颜色条
    cb = fig.colorbar(hb, ax=ax, label='样本密度 (数据点数量)')

    # 设置坐标轴标签
    ax.set_xlabel('归一化速度 (v)', fontsize=14, fontweight='bold')
    ax.set_ylabel('归一化转角 (δ)', fontsize=14, fontweight='bold')

    # 设置标题
    ax.set_title('样本分布热力图\n',
                 fontsize=16, fontweight='bold', pad=20)

    # 添加零线参考
    ax.axhline(y=0, color='white', linestyle='--', linewidth=0.8, alpha=0.7)

    # 设置坐标轴范围和刻度
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(-1, 1.1, 0.2))

    # 添加网格
    ax.grid(True, alpha=0.3, color='white', linestyle='--')

    # 添加注释说明
    # textstr = 'Data Distribution Characteristics:\n'
    # textstr += '• Low speed + High steering: Parking/U-turn\n'
    # textstr += '• Medium speed + Low steering: City driving\n'
    # textstr += '• High speed + Near-zero steering: Highway'

    props = dict(boxstyle='round', facecolor='black', alpha=0.7)
    # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
    #         verticalalignment='top', color='white', bbox=props)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Heatmap saved to: {output_path}")

    plt.close()

    return output_path


def create_histogram_2d(speed, steering, output_path='speed_steering_hist2d.png'):
    """
    创建二维直方图热力图（另一种可视化方式）

    参数:
        speed: 归一化速度数组
        steering: 归一化转角数组
        output_path: 输出图片路径
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    # 使用hist2d绘制二维直方图
    hb = ax.hist2d(
        speed,
        steering,
        bins=80,  # 分箱数量
        cmap='hot',  # 热力图配色
        range=[[0, 1], [-1, 1]]  # 坐标范围
    )

    # 添加颜色条
    cb = fig.colorbar(hb[3], ax=ax, label='Sample Density')

    # 设置坐标轴
    ax.set_xlabel('Normalized Speed (v)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Steering Angle (δ)', fontsize=14, fontweight='bold')
    ax.set_title('Autonomous Driving: Speed vs Steering Angle\n(2D Histogram Heatmap)',
                 fontsize=16, fontweight='bold')

    # 添加参考线
    ax.axhline(y=0, color='cyan', linestyle='--', linewidth=1, alpha=0.7)

    # 设置刻度
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Histogram 2D saved to: {output_path}")

    plt.close()

    return output_path


def analyze_data_distribution(speed, steering):
    """
    分析数据分布特征
    """
    print("\n" + "=" * 60)
    print("Data Distribution Analysis")
    print("=" * 60)

    print(f"\nTotal samples: {len(speed):,}")

    # 速度分析
    print(f"\n[Speed Distribution]")
    print(f"  Mean: {np.mean(speed):.4f}")
    print(f"  Std:  {np.std(speed):.4f}")
    print(f"  Min:  {np.min(speed):.4f}")
    print(f"  Max:  {np.max(speed):.4f}")
    print(f"  Median: {np.median(speed):.4f}")

    # 按速度区间统计
    speed_bins = [
        (0, 0.2, "Very Low (0-0.2)"),
        (0.2, 0.4, "Low (0.2-0.4)"),
        (0.4, 0.6, "Medium (0.4-0.6)"),
        (0.6, 0.8, "High (0.6-0.8)"),
        (0.8, 1.0, "Very High (0.8-1.0)")
    ]

    print(f"\n  Speed Range Distribution:")
    for low, high, label in speed_bins:
        count = np.sum((speed >= low) & (speed < high))
        pct = count / len(speed) * 100
        print(f"    {label}: {count:,} ({pct:.1f}%)")

    # 转角分析
    print(f"\n[Steering Distribution]")
    print(f"  Mean: {np.mean(steering):.4f}")
    print(f"  Std:  {np.std(steering):.4f}")
    print(f"  Min:  {np.min(steering):.4f}")
    print(f"  Max:  {np.max(steering):.4f}")

    # 按转角区间统计
    steering_bins = [
        (-1, -0.5, "Hard Left (-1 to -0.5)"),
        (-0.5, -0.2, "Medium Left (-0.5 to -0.2)"),
        (-0.2, 0.2, "Straight (-0.2 to 0.2)"),
        (0.2, 0.5, "Medium Right (0.2 to 0.5)"),
        (0.5, 1.0, "Hard Right (0.5 to 1.0)")
    ]

    print(f"\n  Steering Range Distribution:")
    for low, high, label in steering_bins:
        count = np.sum((steering >= low) & (steering < high))
        pct = count / len(steering) * 100
        print(f"    {label}: {count:,} ({pct:.1f}%)")

    # 速度与转角的相关性
    correlation = np.corrcoef(speed, steering)[0, 1]
    print(f"\n[Correlation]")
    print(f"  Speed-Steering Correlation: {correlation:.4f}")
    print(f"  (Negative value indicates inverse relationship)")

    print("\n" + "=" * 60)


def main():
    """
    主函数：生成数据并绘制热力图
    """
    print("=" * 60)
    print("Autonomous Driving Data Visualization")
    print("Speed vs Steering Angle Heatmap Generator")
    print("=" * 60)

    # 生成符合自动驾驶分布的合成数据
    print("\n[1/4] Generating synthetic autonomous driving data...")
    speed, steering = generate_autonomous_driving_data(num_samples=50000, seed=42)
    print(f"      Generated {len(speed):,} data points")

    # 分析数据分布
    print("\n[2/4] Analyzing data distribution...")
    analyze_data_distribution(speed, steering)

    # 创建hexbin热力图
    print("\n[3/4] Creating hexbin heatmap...")
    output1 = create_heatmap(speed, steering, 'speed_steering_heatmap.png')

    # 创建2D直方图热力图
    print("\n[4/4] Creating 2D histogram heatmap...")
    output2 = create_histogram_2d(speed, steering, 'speed_steering_hist2d.png')

    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  1. {output1}")
    print(f"  2. {output2}")
    print("\nData points are concentrated where the color is darker.")
    print("The funnel shape shows the safe driving envelope:")
    print("  - High speed → Small steering angles only")
    print("  - Low speed → Full range of steering angles")


if __name__ == "__main__":
    main()
