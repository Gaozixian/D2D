#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹分析和转向决策系统
作者：MiniMax Agent
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import warnings
from typing import Tuple, List, Dict, Any
import os

def setup_matplotlib_for_plotting():
    """
    Setup matplotlib and seaborn for plotting with proper configuration.
    Call this function before creating any plots to ensure proper rendering.
    """
    warnings.filterwarnings('default')  # Show all warnings
    
    # Configure matplotlib for non-interactive mode
    plt.switch_backend("Agg")
    
    # Configure platform-appropriate fonts for cross-platform compatibility
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False

class TrajectoryDataGenerator:
    """轨迹数据生成器"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def generate_sample_trajectory(self, num_points=400) -> pd.DataFrame:
        """
        生成示例轨迹数据
        
        Args:
            num_points: 轨迹点数量
            
        Returns:
            DataFrame: 包含速度、转角、x、y、z的轨迹数据
        """
        # 时间序列
        dt = 0.1  # 时间间隔0.1秒
        time = np.arange(0, num_points * dt, dt)[:num_points]
        
        # 生成轨迹模式：起步 -> 直行 -> 左转 -> 直行 -> 右转 -> 停车
        segments = self._generate_trajectory_segments(num_points)
        
        # 计算位置、速度和转角
        positions = self._calculate_positions(segments, num_points)
        velocities = self._calculate_velocities(positions, dt)
        angles = self._calculate_angles(positions)
        
        # 创建DataFrame
        data = pd.DataFrame({
            'timestamp': time,
            'x': positions[:, 0],
            'y': positions[:, 1], 
            'z': positions[:, 2],
            'velocity': velocities,
            'steering_angle': angles
        })
        
        return data
    
    def _generate_trajectory_segments(self, num_points: int) -> Dict[str, Any]:
        """生成轨迹分段信息"""
        # 分段长度
        segment_sizes = {
            'start': int(num_points * 0.1),    # 起步：10%
            'straight1': int(num_points * 0.2), # 第一段直行：20%
            'left_turn': int(num_points * 0.15), # 左转：15%
            'straight2': int(num_points * 0.25), # 第二段直行：25%
            'right_turn': int(num_points * 0.15), # 右转：15%
            'stop': int(num_points * 0.15)     # 停车：15%
        }
        
        # 确保总数等于num_points
        total = sum(segment_sizes.values())
        if total != num_points:
            segment_sizes['straight2'] += (num_points - total)
            
        return segment_sizes
    
    def _calculate_positions(self, segments: Dict[str, Any], num_points: int) -> np.ndarray:
        """根据分段计算位置"""
        positions = np.zeros((num_points, 3))
        
        current_pos = np.array([0.0, 0.0, 0.0])
        current_index = 0
        
        # 起步段：从静止开始加速
        start_size = segments['start']
        for i in range(start_size):
            speed = 0.5 + 2.0 * (i / start_size)  # 速度从0.5到2.5
            positions[current_index + i] = current_pos + np.array([speed * 0.1, 0, 0])
        current_pos = positions[current_index + start_size - 1]
        current_index += start_size
        
        # 第一段直行
        straight1_size = segments['straight1']
        for i in range(straight1_size):
            speed = 2.5 + 0.5 * np.sin(i * 0.01)  # 轻微速度变化
            positions[current_index + i] = current_pos + np.array([0, speed * 0.1, 0])
        current_pos = positions[current_index + straight1_size - 1]
        current_index += straight1_size
        
        # 左转段
        left_turn_size = segments['left_turn']
        center = current_pos + np.array([5, 0, 0])  # 圆心
        radius = 5.0
        start_angle = 0
        for i in range(left_turn_size):
            angle = start_angle + i * np.pi / (2 * left_turn_size)  # 90度转弯
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            positions[current_index + i] = np.array([x, y, 0])
        current_pos = positions[current_index + left_turn_size - 1]
        current_index += left_turn_size
        
        # 第二段直行
        straight2_size = segments['straight2']
        for i in range(straight2_size):
            speed = 2.0 + 0.3 * np.sin(i * 0.02)
            positions[current_index + i] = current_pos + np.array([speed * 0.1, 0, 0])
        current_pos = positions[current_index + straight2_size - 1]
        current_index += straight2_size
        
        # 右转段
        right_turn_size = segments['right_turn']
        center = current_pos + np.array([0, -5, 0])
        radius = 5.0
        start_angle = -np.pi/2
        for i in range(right_turn_size):
            angle = start_angle + i * np.pi / (2 * right_turn_size)
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            positions[current_index + i] = np.array([x, y, 0])
        current_pos = positions[current_index + right_turn_size - 1]
        current_index += right_turn_size
        
        # 停车段
        stop_size = segments['stop']
        for i in range(stop_size):
            speed = max(0, 2.0 * (1 - i / stop_size))  # 减速到0
            positions[current_index + i] = current_pos + np.array([speed * 0.05, 0, 0])
        
        return positions
    
    def _calculate_velocities(self, positions: np.ndarray, dt: float) -> np.ndarray:
        """计算速度"""
        velocities = np.zeros(len(positions))
        for i in range(1, len(positions)):
            velocity = np.linalg.norm(positions[i] - positions[i-1]) / dt
            velocities[i] = velocity
        return velocities
    
    def _calculate_angles(self, positions: np.ndarray) -> np.ndarray:
        """计算转角"""
        angles = np.zeros(len(positions))
        for i in range(1, len(positions)):
            # 计算方向向量
            direction = positions[i] - positions[i-1]
            # 计算与x轴的夹角
            angle = np.arctan2(direction[1], direction[0])
            angles[i] = angle
        return angles

class TrajectoryAnalyzer:
    """轨迹分析器"""
    
    def __init__(self):
        pass
    
    def match_current_trajectory(self, historical_data: pd.DataFrame, 
                               current_data: pd.DataFrame, 
                               window_size: int = 20) -> Dict[str, Any]:
        """
        匹配当前轨迹与历史轨迹
        
        Args:
            historical_data: 历史轨迹数据
            current_data: 当前轨迹数据
            window_size: 匹配窗口大小
            
        Returns:
            Dict: 匹配结果
        """
        if len(current_data) < window_size:
            return {"match_score": 0, "matched_position": None, "similarity": 0}
        
        # 获取当前轨迹的最后一个窗口
        current_window = current_data.iloc[-window_size:][['x', 'y', 'z']].values
        
        best_match = None
        best_score = 0
        
        # 在历史轨迹中寻找最佳匹配
        for i in range(len(historical_data) - window_size + 1):
            historical_window = historical_data.iloc[i:i+window_size][['x', 'y', 'z']].values
            
            # 计算相似度分数
            score = self._calculate_similarity(current_window, historical_window)
            
            if score > best_score:
                best_score = score
                best_match = i
        
        # 计算匹配位置
        matched_position = None
        if best_match is not None:
            matched_position = historical_data.iloc[best_match + window_size - 1][['x', 'y', 'z']].values
        
        return {
            "match_score": best_score,
            "matched_position": matched_position,
            "match_index": best_match,
            "similarity": best_score
        }
    
    def _calculate_similarity(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """计算两条轨迹的相似度"""
        if len(traj1) != len(traj2):
            return 0
        
        # 计算点对点的欧氏距离
        distances = [euclidean(traj1[i], traj2[i]) for i in range(len(traj1))]
        
        # 转换为相似度分数（距离越小，相似度越高）
        avg_distance = np.mean(distances)
        similarity = 1 / (1 + avg_distance)
        
        return similarity

class TurnClassifier:
    """转向分类器"""
    
    def __init__(self, velocity_threshold: float = 0.5, 
                 angle_threshold: float = 0.3):
        """
        初始化转向分类器
        
        Args:
            velocity_threshold: 速度阈值，用于检测停车/起步
            angle_threshold: 转角阈值，用于检测转向
        """
        self.velocity_threshold = velocity_threshold
        self.angle_threshold = angle_threshold
    
    def classify_trajectory(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        对轨迹进行转向分类
        
        Args:
            data: 轨迹数据
            
        Returns:
            DataFrame: 包含分类结果的轨迹数据
        """
        result_data = data.copy()
        classifications = []
        
        for i in range(len(data)):
            classification = self._classify_single_point(data, i)
            classifications.append(classification)
        
        result_data['action'] = classifications
        return result_data
    
    def _classify_single_point(self, data: pd.DataFrame, index: int) -> str:
        """分类单个轨迹点"""
        if index < 5:
            return "起步"
        
        # 获取当前点和前后窗口的数据
        current_velocity = data.iloc[index]['velocity']
        current_angle = data.iloc[index]['steering_angle']
        
        # 计算角度变化
        angle_change = self._calculate_angle_change(data, index)
        
        # 计算速度变化
        velocity_change = self._calculate_velocity_change(data, index)
        
        # 分类逻辑
        if current_velocity < self.velocity_threshold:
            return "停车"
        elif velocity_change > 0.5 and current_velocity < 1.0:
            return "起步"
        elif abs(angle_change) > self.angle_threshold:
            if angle_change > 0:
                return "左转"
            else:
                return "右转"
        else:
            return "直行"
    
    def _calculate_angle_change(self, data: pd.DataFrame, index: int, window: int = 3) -> float:
        """计算角度变化"""
        if index < window:
            return 0
        
        current_angle = data.iloc[index]['steering_angle']
        prev_angle = data.iloc[index - window]['steering_angle']
        
        # 处理角度跳跃（-π 到 π）
        angle_diff = current_angle - prev_angle
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        return angle_diff
    
    def _calculate_velocity_change(self, data: pd.DataFrame, index: int, window: int = 3) -> float:
        """计算速度变化"""
        if index < window:
            return 0
        
        current_velocity = data.iloc[index]['velocity']
        prev_velocity = data.iloc[index - window]['velocity']
        
        return current_velocity - prev_velocity

class TrajectoryVisualizer:
    """轨迹可视化器"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        setup_matplotlib_for_plotting()
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def plot_trajectory_analysis(self, data: pd.DataFrame, 
                               matched_position: np.ndarray = None) -> str:
        """
        绘制轨迹分析结果
        
        Args:
            data: 轨迹数据
            matched_position: 匹配位置
            
        Returns:
            str: 保存的文件路径
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 颜色映射
        action_colors = {
            '起步': 'green',
            '直行': 'blue', 
            '左转': 'orange',
            '右转': 'red',
            '停车': 'purple'
        }
        
        # 1. 轨迹图
        for action in data['action'].unique():
            mask = data['action'] == action
            ax1.scatter(data[mask]['x'], data[mask]['y'], 
                       c=action_colors[action], label=action, alpha=0.7, s=20)
        
        ax1.set_xlabel('X坐标')
        ax1.set_ylabel('Y坐标')
        ax1.set_title('车辆轨迹及转向分类')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 标注匹配位置
        if matched_position is not None:
            ax1.scatter(matched_position[0], matched_position[1], 
                       c='black', s=100, marker='*', 
                       label='匹配位置', edgecolors='white', linewidth=2)
            ax1.legend()
        
        # 2. 速度曲线
        ax2.plot(data['timestamp'], data['velocity'], 'b-', linewidth=2)
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('速度 (m/s)')
        ax2.set_title('速度变化曲线')
        ax2.grid(True, alpha=0.3)
        
        # 3. 转角曲线
        ax3.plot(data['timestamp'], data['steering_angle'], 'r-', linewidth=2)
        ax3.set_xlabel('时间 (s)')
        ax3.set_ylabel('转角 (rad)')
        ax3.set_title('转角变化曲线')
        ax3.grid(True, alpha=0.3)
        
        # 4. 分类统计
        action_counts = data['action'].value_counts()
        colors = [action_colors[action] for action in action_counts.index]
        ax4.bar(action_counts.index, action_counts.values, color=colors)
        ax4.set_xlabel('行为类别')
        ax4.set_ylabel('点数量')
        ax4.set_title('转向行为统计')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(self.output_dir, 'trajectory_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_trajectory_comparison(self, historical_data: pd.DataFrame, 
                                 current_data: pd.DataFrame, 
                                 match_result: Dict[str, Any]) -> str:
        """绘制轨迹对比图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 历史轨迹
        ax1.plot(historical_data['x'], historical_data['y'], 'b-', linewidth=2, alpha=0.7)
        ax1.scatter(historical_data['x'], historical_data['y'], c='blue', alpha=0.5, s=10)
        ax1.set_xlabel('X坐标')
        ax1.set_ylabel('Y坐标')
        ax1.set_title('历史轨迹')
        ax1.grid(True, alpha=0.3)
        
        # 当前轨迹
        ax2.plot(current_data['x'], current_data['y'], 'r-', linewidth=2, alpha=0.7)
        ax2.scatter(current_data['x'], current_data['y'], c='red', alpha=0.5, s=10)
        
        # 标注匹配位置
        if match_result['matched_position'] is not None:
            ax2.scatter(match_result['matched_position'][0], 
                       match_result['matched_position'][1], 
                       c='green', s=150, marker='*', 
                       label='匹配位置', edgecolors='white', linewidth=2)
            ax2.legend()
        
        ax2.set_xlabel('X坐标')
        ax2.set_ylabel('Y坐标')
        ax2.set_title(f'当前轨迹 (相似度: {match_result["similarity"]:.3f})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'trajectory_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

def main():
    """主函数 - 演示完整的轨迹分析系统"""
    print("🚗 轨迹分析和转向决策系统演示")
    print("=" * 50)
    
    # 1. 生成示例数据
    print("📊 生成示例轨迹数据...")
    generator = TrajectoryDataGenerator()
    
    # 生成历史轨迹（完整轨迹）
    historical_data = generator.generate_sample_trajectory(400)
    
    # 生成当前轨迹（历史轨迹的一部分，用于测试匹配）
    current_data = historical_data.iloc[200:280].copy().reset_index(drop=True)
    
    print(f"✅ 历史轨迹: {len(historical_data)} 个数据点")
    print(f"✅ 当前轨迹: {len(current_data)} 个数据点")
    
    # 2. 轨迹匹配分析
    print("\n🔍 进行轨迹匹配分析...")
    analyzer = TrajectoryAnalyzer()
    match_result = analyzer.match_current_trajectory(
        historical_data, current_data, window_size=20
    )
    
    print(f"✅ 匹配相似度: {match_result['similarity']:.3f}")
    if match_result['matched_position'] is not None:
        pos = match_result['matched_position']
        print(f"✅ 匹配位置: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    # 3. 转向分类
    print("\n🎯 进行转向分类...")
    classifier = TurnClassifier()
    
    # 对历史轨迹进行分类
    historical_classified = classifier.classify_trajectory(historical_data)
    
    # 对当前轨迹进行分类
    current_classified = classifier.classify_trajectory(current_data)
    
    print("✅ 历史轨迹分类完成")
    print("✅ 当前轨迹分类完成")
    
    # 4. 统计结果
    print("\n📈 分类统计结果:")
    print("历史轨迹:")
    for action, count in historical_classified['action'].value_counts().items():
        percentage = count / len(historical_classified) * 100
        print(f"  {action}: {count} 个点 ({percentage:.1f}%)")
    
    print("\n当前轨迹:")
    for action, count in current_classified['action'].value_counts().items():
        percentage = count / len(current_classified) * 100
        print(f"  {action}: {count} 个点 ({percentage:.1f}%)")
    
    # 5. 保存数据
    print("\n💾 保存分析结果...")
    historical_classified.to_csv('output/historical_trajectory_classified.csv', index=False)
    current_classified.to_csv('output/current_trajectory_classified.csv', index=False)
    
    # 6. 生成可视化
    print("\n📊 生成可视化图表...")
    visualizer = TrajectoryVisualizer('output')
    
    # 绘制历史轨迹分析
    hist_plot_path = visualizer.plot_trajectory_analysis(
        historical_classified, match_result['matched_position']
    )
    
    # 绘制轨迹对比
    comparison_plot_path = visualizer.plot_trajectory_comparison(
        historical_data, current_data, match_result
    )
    
    print(f"✅ 轨迹分析图: {hist_plot_path}")
    print(f"✅ 轨迹对比图: {comparison_plot_path}")
    
    # 7. 转向决策分析
    print("\n🎯 当前时刻转向决策分析:")
    if len(current_classified) > 0:
        current_action = current_classified.iloc[-1]['action']
        current_velocity = current_classified.iloc[-1]['velocity']
        current_steering = current_classified.iloc[-1]['steering_angle']
        
        print(f"当前状态: {current_action}")
        print(f"当前速度: {current_velocity:.2f} m/s")
        print(f"当前转角: {current_steering:.2f} rad")
        
        # 预测下一个动作
        next_action_prediction = predict_next_action(current_classified)
        print(f"预测下一步: {next_action_prediction}")
    
    print("\n🎉 轨迹分析系统演示完成！")
    print("\n输出文件:")
    print("- 历史轨迹分类: output/historical_trajectory_classified.csv")
    print("- 当前轨迹分类: output/current_trajectory_classified.csv") 
    print("- 轨迹分析图: output/trajectory_analysis.png")
    print("- 轨迹对比图: output/trajectory_comparison.png")

def predict_next_action(data: pd.DataFrame, window_size: int = 10) -> str:
    """预测下一个动作"""
    if len(data) < window_size:
        return data.iloc[-1]['action'] if len(data) > 0 else "直行"
    
    # 分析最近的动作模式
    recent_actions = data.iloc[-window_size:]['action'].tolist()
    
    # 统计最近动作
    action_counts = {}
    for action in recent_actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    # 如果最近主要是转向，可能继续转向
    if '左转' in action_counts and action_counts['左转'] > window_size * 0.6:
        return "直行"  # 转弯后通常直行
    elif '右转' in action_counts and action_counts['右转'] > window_size * 0.6:
        return "直行"
    elif action_counts.get('停车', 0) > window_size * 0.7:
        return "起步"
    else:
        return "直行"  # 默认预测直行

if __name__ == "__main__":
    main()