#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双路径对比可视化工具
支持读取两个CSV文件并绘制到同一个图窗中进行对比

使用方法：
python plot_two_paths.py path1.csv path2.csv
python plot_two_paths.py path1.csv path2.csv --output comparison.png

作者：MiniMax Agent
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import os

# 设置中文字体支持
matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 使用黑体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 颜色方案
PATH_COLORS = {
    'path1': '#2E86AB',  # 蓝色
    'path2': '#E94F37',  # 红色
}

MARKERS = ['o', 's', '^', 'D', 'v', 'p']


def load_csv_data(csv_file, label="文件"):
    """
    加载CSV文件并验证数据格式
    Args:
        csv_file: CSV文件路径
        label: 文件标签（用于显示）

    Returns:
        DataFrame: 加载的数据
    """
    if not os.path.exists(csv_file):
        print(f"❌ 错误：{label}不存在 - {csv_file}")
        return None

    try:
        df = pd.read_csv(csv_file)
        print(f"✅ 成功加载{label}: {csv_file}")
        print(f"   数据点数量: {len(df)}")
        print(f"   列名: {list(df.columns)}")

        # 验证必需的列
        required_cols = ['global_x', 'global_y']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"❌ 缺少必需的列: {missing_cols}")
            return None

        # 显示数据统计
        print(f"\n📊 {label}数据统计:")
        print(f"   X范围: [{df['global_x'].min():.2f}, {df['global_x'].max():.2f}]")
        print(f"   Y范围: [{df['global_y'].min():.2f}, {df['global_y'].max():.2f}]")

        total_length = calculate_path_length(df)
        print(f"   路径总长: {total_length:.2f} m")

        if 'z' in df.columns:
            print(f"   Z范围: [{df['global_z'].min():.2f}, {df['global_z'].max():.2f}]")

        return df

    except Exception as e:
        print(f"❌ 加载{label}失败: {str(e)}")
        return None


def calculate_path_length(df):
    """计算路径总长度"""
    total_length = 0.0

    for i in range(1, len(df)):
        dx = df['global_x'].iloc[i] - df['global_x'].iloc[i-1]
        dy = df['global_y'].iloc[i] - df['global_y'].iloc[i-1]

        if 'global_z' in df.columns:
            dz = df['global_z'].iloc[i] - df['global_z'].iloc[i-1]
            segment_length = np.sqrt(dx**2 + dy**2 + dz**2)
        else:
            segment_length = np.sqrt(dx**2 + dy**2)

        total_length += segment_length

    return total_length


def plot_two_paths_2d(df1, df2, label1="reference_path", label2="recorded_path",
                      output_file="two_paths_2d.png", title="双路径对比 2D视图",
                      figsize=(14, 10), point_size=30, line_width=2.5,
                      show_numbers=False, alpha_path=0.7):
    """
    绘制2D双路径对比图

    Args:
        df1: 第一个路径数据DataFrame
        df2: 第二个路径数据DataFrame
        label1: 第一个路径的标签
        label2: 第二个路径的标签
        output_file: 输出文件名
        title: 图表标题
        figsize: 图表大小
        point_size: 点的大小
        line_width: 连线宽度
        show_numbers: 是否显示点编号
        alpha_path: 路径透明度
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制reference_path
    ax.plot(df1['global_x'], df1['global_y'], color=PATH_COLORS['path1'], linewidth=line_width,
            alpha=alpha_path, label=label1, linestyle='-')
    ax.scatter(df1['global_x'], df1['global_y'], c=[PATH_COLORS['path1']], s=point_size,
              edgecolors='white', linewidth=0.5, zorder=5, alpha=0.8)

    # 绘制recorded_path
    ax.plot(df2['global_x'], df2['global_y'], color=PATH_COLORS['path2'], linewidth=line_width,
            alpha=alpha_path, label=label2, linestyle='--')
    ax.scatter(df2['global_x'], df2['global_y'], c=[PATH_COLORS['path2']], s=point_size,
              edgecolors='white', linewidth=0.5, zorder=5, alpha=0.8)

    # 标记起点和终点
    # reference_path
    ax.scatter(df1['global_x'].iloc[0], df1['global_y'].iloc[0], c='green', s=200,
              marker='o', zorder=10, edgecolors='black', linewidth=2)
    ax.scatter(df1['global_x'].iloc[-1], df1['global_y'].iloc[-1], c='darkgreen', s=200,
              marker='s', zorder=10, edgecolors='black', linewidth=2)

    # recorded_path
    ax.scatter(df2['global_x'].iloc[0], df2['global_y'].iloc[0], c='orange', s=200,
              marker='o', zorder=10, edgecolors='black', linewidth=2)
    ax.scatter(df2['global_x'].iloc[-1], df2['global_y'].iloc[-1], c='darkorange', s=200,
              marker='s', zorder=10, edgecolors='black', linewidth=2)

    # 添加图例（带起点终点说明）
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=PATH_COLORS['path1'], linewidth=2, label=f'{label1}'),
        Line2D([0], [0], color=PATH_COLORS['path2'], linewidth=2, linestyle='--', label=f'{label2}'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=12,
               markeredgecolor='black', label=f'{label1}起点'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='darkgreen', markersize=12,
               markeredgecolor='black', label=f'{label1}终点'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange', markersize=12,
               markeredgecolor='black', label=f'{label2}起点'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='darkorange', markersize=12,
               markeredgecolor='black', label=f'{label2}终点'),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)

    # 显示点编号（可选）
    if show_numbers:
        for i, (x, y) in enumerate(zip(df1['global_x'], df1['global_y'])):
            if i % 10 == 0:  # 每10个点显示一次
                ax.annotate(str(i), (x, y), textcoords="offset points",
                           xytext=(5, 5), fontsize=7, alpha=0.7, color=PATH_COLORS['path1'])

        for i, (x, y) in enumerate(zip(df2['global_x'], df2['global_y'])):
            if i % 10 == 0:
                ax.annotate(str(i), (x, y), textcoords="offset points",
                           xytext=(5, -10), fontsize=7, alpha=0.7, color=PATH_COLORS['path2'])

    # 设置图表属性
    ax.set_xlabel('X 坐标 (m)', fontsize=12)
    ax.set_ylabel('Y 坐标 (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # 添加统计信息
    len1 = calculate_path_length(df1)
    len2 = calculate_path_length(df2)

    stats_text = (f"{label1}:\n"
                  f"Point: {len(df1)}\n"
                  f"Length: {len1:.1f} m\n\n"
                  f"{label2}:\n"
                  f"Point: {len(df2)}\n"
                  f"Length: {len2:.1f} m")

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
           family='monospace')

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 设置中文显示

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 2D对比图已保存: {output_file}")
    return output_file


def plot_two_paths_3d(df1, df2, label1="reference_path", label2="recorded_path",
                      output_file="two_paths_3d.png", title="双路径对比 3D视图",
                      figsize=(14, 10), point_size=30, line_width=2.5):
    """
    绘制3D双路径对比图

    Args:
        df1: 第一个路径数据DataFrame
        df2: 第二个路径数据DataFrame
        label1: 第一个路径的标签
        label2: 第二个路径的标签
        output_file: 输出文件名
        title: 图表标题
        figsize: 图表大小
        point_size: 点的大小
        line_width: 连线宽度
    """
    # 确保有Z列
    df1 = df1.copy()
    df2 = df2.copy()

    if 'z' not in df1.columns:
        df1['z'] = np.zeros(len(df1))
    if 'z' not in df2.columns:
        df2['z'] = np.zeros(len(df2))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # 绘制reference_path
    ax.plot(df1['global_x'], df1['global_y'], df1['z'], color=PATH_COLORS['path1'],
            linewidth=line_width, alpha=0.8, label=label1)
    ax.scatter(df1['global_x'], df1['global_y'], df1['z'], c=[PATH_COLORS['path1']],
              s=point_size, edgecolors='white', linewidth=0.5)

    # 绘制recorded_path
    ax.plot(df2['global_x'], df2['global_y'], df2['z'], color=PATH_COLORS['path2'],
            linewidth=line_width, linestyle='--', alpha=0.8, label=label2)
    ax.scatter(df2['global_x'], df2['global_y'], df2['z'], c=[PATH_COLORS['path2']],
              s=point_size, edgecolors='white', linewidth=0.5)

    # 标记起点和终点
    ax.scatter(df1['global_x'].iloc[0], df1['global_y'].iloc[0], df1['z'].iloc[0],
              c='green', s=150, marker='o', label=f'{label1}起点')
    ax.scatter(df1['global_x'].iloc[-1], df1['global_y'].iloc[-1], df1['z'].iloc[-1],
              c='darkgreen', s=150, marker='s', label=f'{label1}终点')
    ax.scatter(df2['global_x'].iloc[0], df2['global_y'].iloc[0], df2['z'].iloc[0],
              c='orange', s=150, marker='o', label=f'{label2}起点')
    ax.scatter(df2['global_x'].iloc[-1], df2['global_y'].iloc[-1], df2['z'].iloc[-1],
              c='darkorange', s=150, marker='s', label=f'{label2}终点')

    # 设置标签
    ax.set_xlabel('X 坐标 (m)', fontsize=10)
    ax.set_ylabel('Y 坐标 (m)', fontsize=10)
    ax.set_zlabel('Z 坐标 (m)', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)

    # 添加统计信息
    len1 = calculate_path_length(df1)
    len2 = calculate_path_length(df2)

    stats_text = (f"{label1}: {len1:.1f} m\n"
                  f"{label2}: {len2:.1f} m")

    ax.text2D(0.02, 0.95, stats_text, transform=ax.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
             family='monospace')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 3D对比图已保存: {output_file}")
    return output_file


def plot_path_deviation(df1, df2, output_file="path_deviation.png",
                       title="路径偏差分析", figsize=(14, 10)):
    """
    计算并绘制两条路径之间的偏差

    Args:
        df1: 第一个路径数据DataFrame
        df2: 第二个路径数据DataFrame
        output_file: 输出文件名
        title: 图表标题
        figsize: 图表大小
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. 左上：路径对比图
    ax1 = axes[0, 0]
    ax1.plot(df1['global_x'], df1['global_y'], color=PATH_COLORS['path1'], linewidth=2,
             alpha=0.8, label='参考路径')
    ax1.plot(df2['global_x'], df2['global_y'], color=PATH_COLORS['path2'], linewidth=2,
             linestyle='--', alpha=0.8, label='对比路径')
    ax1.scatter(df1['global_x'].iloc[0], df1['global_y'].iloc[0], c='green', s=100,
               marker='o', zorder=10)
    ax1.scatter(df1['global_x'].iloc[-1], df1['global_y'].iloc[-1], c='darkgreen', s=100,
               marker='s', zorder=10)
    ax1.set_xlabel('X 坐标 (m)')
    ax1.set_ylabel('Y 坐标 (m)')
    ax1.set_title('路径对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 2. 右上：点到点的最短距离
    ax2 = axes[0, 1]

    # 计算每个点的最近距离
    deviations = []
    for i in range(min(len(df1), len(df2))):
        point1 = (df1['global_x'].iloc[i], df1['global_y'].iloc[i])
        point2 = (df2['global_x'].iloc[i], df2['global_y'].iloc[i])
        dist = np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
        deviations.append(dist)

    ax2.plot(range(len(deviations)), deviations, 'b-', linewidth=2)
    ax2.fill_between(range(len(deviations)), deviations, alpha=0.3)
    ax2.axhline(y=np.mean(deviations), color='red', linestyle='--',
               label=f'平均偏差: {np.mean(deviations):.2f} m')
    ax2.axhline(y=np.max(deviations), color='orange', linestyle=':',
               label=f'最大偏差: {np.max(deviations):.2f} m')
    ax2.set_xlabel('点序号')
    ax2.set_ylabel('偏差距离 (m)')
    ax2.set_title('路径偏差分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 左下：X坐标对比
    ax3 = axes[1, 0]
    x_range = range(min(len(df1), len(df2)))
    ax3.plot(x_range, df1['global_x'].iloc[:len(x_range)], color=PATH_COLORS['path1'],
             linewidth=2, label='参考路径')
    ax3.plot(x_range, df2['global_x'].iloc[:len(x_range)], color=PATH_COLORS['path2'],
             linewidth=2, linestyle='--', label='对比路径')
    ax3.set_xlabel('点序号')
    ax3.set_ylabel('X 坐标 (m)')
    ax3.set_title('X坐标对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 右下：Y坐标对比
    ax4 = axes[1, 1]
    ax4.plot(x_range, df1['global_y'].iloc[:len(x_range)], color=PATH_COLORS['path1'],
             linewidth=2, label='参考路径')
    ax4.plot(x_range, df2['global_y'].iloc[:len(x_range)], color=PATH_COLORS['path2'],
             linewidth=2, linestyle='--', label='对比路径')
    ax4.set_xlabel('点序号')
    ax4.set_ylabel('Y 坐标 (m)')
    ax4.set_title('Y坐标对比')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    # 输出统计信息
    avg_deviation = np.mean(deviations)
    max_deviation = np.max(deviations)
    min_deviation = np.min(deviations)

    print(f"✅ 偏差分析图已保存: {output_file}")
    print(f"   平均偏差: {avg_deviation:.3f} m")
    print(f"   最大偏差: {max_deviation:.3f} m")
    print(f"   最小偏差: {min_deviation:.3f} m")

    return output_file


def plot_comparison_with_time(df1, df2, output_file="paths_with_time.png",
                             title="路径与速度时间序列对比", figsize=(16, 10)):
    """
    绘制带时间序列的路径对比图（如果有时间数据）

    Args:
        df1: 第一个路径数据DataFrame
        df2: 第二个路径数据DataFrame
        output_file: 输出文件名
        title: 图表标题
        figsize: 图表大小
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. 左上：路径图
    ax1 = axes[0, 0]
    ax1.plot(df1['global_x'], df1['global_y'], color=PATH_COLORS['path1'], linewidth=2,
             alpha=0.8, label='reference_path')
    ax1.plot(df2['global_x'], df2['global_y'], color=PATH_COLORS['path2'], linewidth=2,
             linestyle='--', alpha=0.8, label='recorded_path')
    ax1.set_xlabel('X 坐标 (m)')
    ax1.set_ylabel('Y 坐标 (m)')
    ax1.set_title('路径对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 2. 右上：X坐标时间序列
    ax2 = axes[0, 1]
    if 'timestamp' in df1.columns:
        time1 = df1['timestamp']
        time2 = df2['timestamp']
        xlabel = '时间 (s)'
    else:
        time1 = range(len(df1))
        time2 = range(len(df2))
        xlabel = '点序号'

    ax2.plot(time1, df1['global_x'], color=PATH_COLORS['path1'], linewidth=2, label='reference_path X')
    ax2.plot(time2, df2['global_x'], color=PATH_COLORS['path2'], linewidth=2, linestyle='--', label='recorded_path X')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('X 坐标 (m)')
    ax2.set_title('X坐标随时间变化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 左下：Y坐标时间序列
    ax3 = axes[1, 0]
    ax3.plot(time1, df1['global_y'], color=PATH_COLORS['path1'], linewidth=2, label='reference_path Y')
    ax3.plot(time2, df2['global_y'], color=PATH_COLORS['path2'], linewidth=2, linestyle='--', label='recorded_path Y')
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel('Y 坐标 (m)')
    ax3.set_title('Y坐标随时间变化')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 右下：速度对比（如果有速度数据）
    ax4 = axes[1, 1]
    if 'velocity' in df1.columns and 'velocity' in df2.columns:
        ax4.plot(time1, df1['velocity'], color=PATH_COLORS['path1'], linewidth=2, label='reference_path 速度')
        ax4.plot(time2, df2['velocity'], color=PATH_COLORS['path2'], linewidth=2, linestyle='--', label='recorded_path 速度')
        ax4.set_ylabel('速度 (m/s)')
    else:
        ax4.text(0.5, 0.5, '无可用速度数据', ha='center', va='center', fontsize=12)
        ax4.set_ylabel('速度')

    ax4.set_xlabel(xlabel)
    ax4.set_title('速度对比')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 时间序列对比图已保存: {output_file}")
    return output_file


def generate_comprehensive_report(df1, df2, label1="reference_path", label2="recorded_path", output_dir="."):
    """
    生成综合分析报告

    Args:
        df1: 第一个路径数据DataFrame
        df2: 第二个路径数据DataFrame
        label1: 第一个路径的标签
        label2: 第二个路径的标签
        output_dir: 输出目录
    """
    print("\n📊 生成综合分析报告...")

    results = {}

    # 1. 2D对比图
    results['2d_comparison'] = plot_two_paths_2d(
        df1, df2, label1, label2,
        os.path.join(output_dir, "comparison_2d.png")
    )

    # 2. 3D对比图
    results['3d_comparison'] = plot_two_paths_3d(
        df1, df2, label1, label2,
        os.path.join(output_dir, "comparison_3d.png")
    )

    # 3. 偏差分析图
    results['deviation'] = plot_path_deviation(
        df1, df2,
        os.path.join(output_dir, "deviation_analysis.png")
    )

    # 4. 时间序列对比图
    results['time_series'] = plot_comparison_with_time(
        df1, df2,
        os.path.join(output_dir, "time_series_comparison.png")
    )

    # 5. 生成统计报告
    len1 = calculate_path_length(df1)
    len2 = calculate_path_length(df2)

    # 计算平均偏差
    deviations = []
    min_len = min(len(df1), len(df2))
    for i in range(min_len):
        dist = np.sqrt((df1['global_x'].iloc[i] - df2['global_x'].iloc[i])**2 +
                       (df1['global_y'].iloc[i] - df2['global_y'].iloc[i])**2)
        deviations.append(dist)

    avg_deviation = np.mean(deviations) if deviations else 0
    max_deviation = np.max(deviations) if deviations else 0

    stats = {
        f'{label1}': {
            '数据点数': len(df1),
            '路径长度': f"{len1:.2f} m",
            'X范围': f"[{df1['global_x'].min():.2f}, {df1['global_x'].max():.2f}]",
            'Y范围': f"[{df1['global_y'].min():.2f}, {df1['global_y'].max():.2f}]",
        },
        f'{label2}': {
            '数据点数': len(df2),
            '路径长度': f"{len2:.2f} m",
            'X范围': f"[{df2['global_x'].min():.2f}, {df2['global_x'].max():.2f}]",
            'Y范围': f"[{df2['global_y'].min():.2f}, {df2['global_y'].max():.2f}]",
        },
        '路径对比': {
            '平均偏差': f"{avg_deviation:.3f} m",
            '最大偏差': f"{max_deviation:.3f} m",
            '长度差异': f"{abs(len1 - len2):.2f} m",
            '差异百分比': f"{abs(len1 - len2) / max(len1, len2) * 100:.1f}%" if max(len1, len2) > 0 else "0%",
        }
    }

    # 保存统计报告
    stats_file = os.path.join(output_dir, "comparison_statistics.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("双路径对比分析报告\n")
        f.write("=" * 60 + "\n\n")

        for section, data in stats.items():
            f.write(f"\n【{section}】\n")
            f.write("-" * 40 + "\n")
            for key, value in data.items():
                f.write(f"  {key}: {value}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("生成的可视化文件:\n")
        f.write("=" * 60 + "\n")

        for name, filepath in results.items():
            if filepath:
                f.write(f"  - {name}: {filepath}\n")

    print(f"✅ 统计报告已保存: {stats_file}")

    return results, stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='双路径对比可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python plot_two_paths.py path1.csv path2.csv
  python plot_two_paths.py path1.csv path2.csv --output comparison.png
  python plot_two_paths.py path1.csv path2.csv --label1 "全局规划" --label2 "实际行驶"
  python plot_two_paths.py path1.csv path2.csv --show-numbers
        """
    )

    parser.add_argument('--csv1', default='global_vehicle_data.csv', help='第一个CSV文件路径')
    parser.add_argument('--csv2', default='ego_vehicle_data.csv', help='第二个CSV文件路径')
    parser.add_argument('--output', '-o', default='comparison.png',
                       help='输出图片文件名 (默认: comparison.png)')
    parser.add_argument('--label1', '-l1', default='reference_path',
                       help='第一个路径的标签 (默认: reference_path)')
    parser.add_argument('--label2', '-l2', default='recorded_path',
                       help='第二个路径的标签 (默认: recorded_path)')
    parser.add_argument('--show-numbers', '-n', action='store_true',
                       help='显示点编号')
    parser.add_argument('--output-dir', '-d', default='.',
                       help='输出目录 (默认: 当前目录)')
    parser.add_argument('--title', '-t', default='路径对比分析',
                       help='图表标题')

    args = parser.parse_args()

    print("🛣️  双路径对比可视化工具")
    print("=" * 60)

    # 加载两个CSV文件
    print(f"\n📂 加载数据文件...")
    df1 = load_csv_data('global_vehicle_data.csv', "文件1")
    df2 = load_csv_data('ego_vehicle_data.csv', "文件2")

    if df1 is None or df2 is None:
        print("❌ 加载数据失败，程序退出")
        return

    # 创建输出目录
    if args.output_dir != '.' and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"✅ 创建输出目录: {args.output_dir}")

    # 生成可视化
    print(f"\n🎨 生成可视化图表...")

    # 如果指定了单个输出文件，生成综合对比图
    if args.output.endswith('.png'):
        output_file = os.path.join(args.output_dir, args.output)
        plot_two_paths_2d(
            df1, df2,
            label1=args.label1,
            label2=args.label2,
            output_file=output_file,
            title=args.title,
            show_numbers=args.show_numbers
        )
        print(f"\n✅ 对比图已保存: {output_file}")
    else:
        # 生成完整的分析报告
        results, stats = generate_comprehensive_report(
            df1, df2,
            label1=args.label1,
            label2=args.label2,
            output_dir=args.output_dir
        )

    print("\n" + "=" * 60)
    print("✅ 可视化完成！")
    print("=" * 60)

    # 输出统计摘要
    len1 = calculate_path_length(df1)
    len2 = calculate_path_length(df2)
    print(f"\n📊 路径统计摘要:")
    print(f"   {args.label1}: {len(df1)} 点, {len1:.2f} m")
    print(f"   {args.label2}: {len(df2)} 点, {len2:.2f} m")
    print(f"   长度差异: {abs(len1 - len2):.2f} m ({abs(len1 - len2) / max(len1, len2) * 100:.1f}%)")


if __name__ == "__main__":
    main()
