#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于全局路径点的车辆转向决策系统
专门针对端到端网络的路径偏差和转向判断问题

作者：MiniMax Agent
功能：
1. 路径点加载和预处理
2. 实时位置跟踪和路径匹配
3. 智能转向决策（直行、左转、右转、停止）
4. 转向角度阈值自适应调整
5. 路径几何分析

特性：
- 欧氏距离匹配最近路径点
- 自适应转向阈值（大角度vs小角度）
- 路径预瞄功能
- 轨迹平滑处理
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from typing import List, Tuple, Dict, Optional
import warnings
import math

def setup_matplotlib_for_plotting():
    """设置matplotlib绘图环境"""
    warnings.filterwarnings('default')
    plt.switch_backend("Agg")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False

class PathPoint:
    """路径点类"""
    def __init__(self, x: float, y: float, index: int = 0):
        self.x = x
        self.y = y
        self.index = index
        self.distance_to_next = 0.0
        self.heading_angle = 0.0

class GlobalPathLoader:
    """全局路径加载器"""
    
    def __init__(self, path_points: List[PathPoint]):
        self.path_points = path_points
        self._preprocess_path()
    
    @classmethod
    def from_csv(cls, csv_file: str) -> 'GlobalPathLoader':
        """从CSV文件加载路径点"""
        try:
            df = pd.read_csv(csv_file)
            # 假设CSV包含x, y列
            if 'x' not in df.columns or 'y' not in df.columns:
                raise ValueError("CSV文件必须包含x和y列")
            
            path_points = []
            for i, row in df.iterrows():
                path_points.append(PathPoint(row['x'], row['y'], i))
            
            return cls(path_points)
        except Exception as e:
            raise Exception(f"加载路径文件失败: {str(e)}")
    
    @classmethod
    def from_coordinates(cls, x_coords: List[float], y_coords: List[float]) -> 'GlobalPathLoader':
        """从坐标列表创建路径"""
        if len(x_coords) != len(y_coords):
            raise ValueError("x和y坐标长度必须相同")
        
        path_points = []
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            path_points.append(PathPoint(x, y, i))
        
        return cls(path_points)
    
    def _preprocess_path(self):
        """预处理路径点"""
        if len(self.path_points) < 2:
            return
        
        # 计算相邻点之间的距离
        for i in range(len(self.path_points) - 1):
            p1 = self.path_points[i]
            p2 = self.path_points[i + 1]
            p1.distance_to_next = euclidean([p1.x, p1.y], [p2.x, p2.y])
        
        # 计算每个点的朝向角
        for i in range(len(self.path_points)):
            if i < len(self.path_points) - 1:
                p1 = self.path_points[i]
                p2 = self.path_points[i + 1]
                p1.heading_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
            else:
                # 最后一个点，使用前一个点的朝向
                if i > 0:
                    self.path_points[i].heading_angle = self.path_points[i-1].heading_angle
    
    def get_point_by_index(self, index: int) -> Optional[PathPoint]:
        """根据索引获取路径点"""
        if 0 <= index < len(self.path_points):
            return self.path_points[index]
        return None
    
    def get_total_length(self) -> float:
        """获取路径总长度"""
        return sum(p.distance_to_next for p in self.path_points[:-1])

class VehicleState:
    """车辆状态类"""
    def __init__(self, x: float, y: float, velocity: float = 0.0, 
                 heading: float = 0.0, timestamp: float = 0.0):
        self.x = x
        self.y = y
        self.velocity = velocity  # m/s
        self.heading = heading    # 弧度
        self.timestamp = timestamp
        self.nearest_path_index = -1
        self.distance_to_path = float('inf')
        self.steering_angle = 0.0

class PathMatcher:
    """路径匹配器"""
    
    def __init__(self, path_loader: GlobalPathLoader, max_search_radius: float = 10.0):
        self.path_loader = path_loader
        self.max_search_radius = max_search_radius
        self.last_matched_index = -1
    
    def find_nearest_point(self, vehicle_state: VehicleState) -> Tuple[Optional[PathPoint], float]:
        """找到距离车辆最近的路径点"""
        min_distance = float('inf')
        nearest_point = None
        nearest_index = -1
        
        # 搜索范围优化：如果有上次匹配结果，只搜索附近区域
        search_start = max(0, self.last_matched_index - 50)
        search_end = min(len(self.path_loader.path_points), self.last_matched_index + 100)
        
        if self.last_matched_index == -1:  # 第一次搜索，搜索全部
            search_start = 0
            search_end = len(self.path_loader.path_points)
        
        for i in range(search_start, search_end):
            path_point = self.path_loader.path_points[i]
            distance = euclidean([vehicle_state.x, vehicle_state.y], [path_point.x, path_point.y])
            
            if distance < min_distance:
                min_distance = distance
                nearest_point = path_point
                nearest_index = i
        
        # 如果距离超过搜索半径，可能需要扩大搜索范围
        if min_distance > self.max_search_radius:
            # 在全路径中搜索
            for i, path_point in enumerate(self.path_loader.path_points):
                distance = euclidean([vehicle_state.x, vehicle_state.y], [path_point.x, path_point.y])
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = path_point
                    nearest_index = i
        
        self.last_matched_index = nearest_index
        vehicle_state.nearest_path_index = nearest_index
        vehicle_state.distance_to_path = min_distance
        
        return nearest_point, min_distance

class SteeringDecisionMaker:
    """转向决策器"""
    
    def __init__(self, max_velocity_ms: float = 15.0/3.6,  # 15km/h 转换为 m/s
                 lookahead_distance: float = 8.0,
                 small_turn_threshold: float = 5.0,      # 小转向角度阈值（度）
                 large_turn_threshold: float = 15.0,     # 大转向角度阈值（度）
                 stop_threshold: float = 0.5):           # 停车阈值 (m/s)
        self.max_velocity_ms = max_velocity_ms
        self.lookahead_distance = lookahead_distance
        self.small_turn_threshold = math.radians(small_turn_threshold)
        self.large_turn_threshold = math.radians(large_turn_threshold)
        self.stop_threshold = stop_threshold
        
        # 历史状态用于平滑决策
        self.decision_history = []
        self.max_history_length = 5
    
    def make_decision(self, vehicle_state: VehicleState, 
                     path_loader: GlobalPathLoader,
                     path_matcher: PathMatcher) -> Dict:
        """做出转向决策"""
        
        # 1. 找到最近的路径点
        nearest_point, distance = path_matcher.find_nearest_point(vehicle_state)
        
        if nearest_point is None:
            return {
                'action': '未知',
                'confidence': 0.0,
                'reason': '无法找到路径点',
                'recommended_speed': 0.0,
                'steering_angle': 0.0
            }
        
        # 2. 计算路径前瞻点
        lookahead_point, lookahead_index = self._find_lookahead_point(
            vehicle_state, path_loader, path_matcher
        )
        
        # 3. 计算转向角度
        steering_angle = self._calculate_steering_angle(
            vehicle_state, nearest_point, lookahead_point
        )
        
        # 4. 判断转向行为
        action = self._classify_steering_action(
            vehicle_state, steering_angle, nearest_point, lookahead_point
        )
        
        # 5. 决策平滑
        smoothed_action = self._smooth_decision(action)
        
        # 6. 计算推荐速度
        recommended_speed = self._calculate_recommended_speed(
            vehicle_state, smoothed_action, steering_angle, distance
        )
        
        # 7. 计算置信度
        confidence = self._calculate_confidence(
            vehicle_state, distance, steering_angle, path_loader
        )
        
        decision = {
            'action': smoothed_action,
            'confidence': confidence,
            'steering_angle': math.degrees(steering_angle),  # 转换为度便于查看
            'recommended_speed': recommended_speed,
            'nearest_distance': distance,
            'lookahead_index': lookahead_index,
            'vehicle_heading': math.degrees(vehicle_state.heading),
            'path_heading': math.degrees(nearest_point.heading_angle),
            'heading_error': math.degrees(steering_angle)
        }
        
        return decision
    
    def _find_lookahead_point(self, vehicle_state: VehicleState,
                            path_loader: GlobalPathLoader,
                            path_matcher: PathMatcher) -> Tuple[Optional[PathPoint], int]:
        """找到前瞻路径点"""
        if vehicle_state.nearest_path_index == -1:
            return None, -1
        
        current_index = vehicle_state.nearest_path_index
        accumulated_distance = 0.0
        target_distance = max(self.lookahead_distance, vehicle_state.velocity * 2.0)  # 至少2秒的前瞻
        
        # 向前搜索到目标距离
        for i in range(current_index, len(path_loader.path_points) - 1):
            accumulated_distance += path_loader.path_points[i].distance_to_next
            
            if accumulated_distance >= target_distance:
                return path_loader.path_points[i], i
        
        # 如果没找到足够远的点，返回路径终点
        return path_loader.path_points[-1], len(path_loader.path_points) - 1
    
    def _calculate_steering_angle(self, vehicle_state: VehicleState,
                                nearest_point: PathPoint,
                                lookahead_point: Optional[PathPoint]) -> float:
        """计算转向角"""
        if lookahead_point is None:
            return 0.0
        
        # 计算车辆朝向与目标方向的角度差
        target_heading = math.atan2(
            lookahead_point.y - vehicle_state.y,
            lookahead_point.x - vehicle_state.x
        )
        
        # 计算角度差，处理跳跃
        angle_diff = target_heading - vehicle_state.heading
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        return angle_diff
    
    def _classify_steering_action(self, vehicle_state: VehicleState,
                                steering_angle: float,
                                nearest_point: PathPoint,
                                lookahead_point: Optional[PathPoint]) -> str:
        """分类转向行为"""
        
        # 速度判断
        if vehicle_state.velocity < self.stop_threshold:
            return '停车'
        
        # 转向角度判断
        abs_angle = abs(steering_angle)
        
        if abs_angle < math.radians(2.0):  # 小于2度认为直行
            return '直行'
        elif abs_angle < self.small_turn_threshold:
            # 小转向：根据方向确定
            if steering_angle > 0:
                return '小左转'
            else:
                return '小右转'
        elif abs_angle < self.large_turn_threshold:
            # 中等转向
            if steering_angle > 0:
                return '左转'
            else:
                return '右转'
        else:
            # 大转向
            if steering_angle > 0:
                return '大左转'
            else:
                return '大右转'
    
    def _smooth_decision(self, current_action: str) -> str:
        """决策平滑"""
        self.decision_history.append(current_action)
        
        if len(self.decision_history) > self.max_history_length:
            self.decision_history.pop(0)
        
        # 简单的多数投票平滑
        if len(self.decision_history) >= 3:
            recent_actions = self.decision_history[-3:]
            action_counts = {}
            for action in recent_actions:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            # 返回出现次数最多的动作
            return max(action_counts.items(), key=lambda x: x[1])[0]
        
        return current_action
    
    def _calculate_recommended_speed(self, vehicle_state: VehicleState,
                                   action: str, steering_angle: float,
                                   path_distance: float) -> float:
        """计算推荐速度"""
        base_speed = self.max_velocity_ms
        
        # 根据转向角度调整速度
        abs_angle = abs(steering_angle)
        if abs_angle > self.large_turn_threshold:
            base_speed *= 0.3  # 大转向减速
        elif abs_angle > self.small_turn_threshold:
            base_speed *= 0.6  # 中等转向减速
        elif abs_angle > math.radians(5.0):
            base_speed *= 0.8  # 小转向轻微减速
        
        # 根据路径距离调整（距离远可以更快）
        if path_distance < 1.0:
            base_speed *= 0.5  # 离路径太近，减速
        
        return min(base_speed, self.max_velocity_ms)
    
    def _calculate_confidence(self, vehicle_state: VehicleState,
                            path_distance: float, steering_angle: float,
                            path_loader: GlobalPathLoader) -> float:
        """计算决策置信度"""
        confidence = 1.0
        
        # 距离置信度：距离越近置信度越高
        if path_distance < 0.5:
            confidence *= 1.0
        elif path_distance < 2.0:
            confidence *= 0.8
        elif path_distance < 5.0:
            confidence *= 0.6
        else:
            confidence *= 0.3
        
        # 速度置信度：速度合适置信度越高
        if vehicle_state.velocity < 0.1:
            confidence *= 0.7  # 停车状态置信度稍低
        elif vehicle_state.velocity > self.max_velocity_ms * 1.2:
            confidence *= 0.5  # 超速置信度低
        else:
            confidence *= 1.0
        
        # 转向角度合理性：过大的转向角可能不可信
        abs_angle = abs(steering_angle)
        if abs_angle > math.radians(45):  # 超过45度
            confidence *= 0.5
        elif abs_angle > math.radians(30):  # 超过30度
            confidence *= 0.7
        
        return max(0.0, min(1.0, confidence))

class PathFollowingSystem:
    """路径跟随系统"""
    
    def __init__(self, path_loader: GlobalPathLoader):
        self.path_loader = path_loader
        self.path_matcher = PathMatcher(path_loader)
        self.decision_maker = SteeringDecisionMaker()
        self.vehicle_states = []
        self.decision_history = []
    
    def update_vehicle_state(self, x: float, y: float, velocity: float = 0.0,
                           heading: float = 0.0, timestamp: float = 0.0) -> Dict:
        """更新车辆状态并做出决策"""
        vehicle_state = VehicleState(x, y, velocity, heading, timestamp)
        
        # 做出决策
        decision = self.decision_maker.make_decision(
            vehicle_state, self.path_loader, self.path_matcher
        )
        
        # 记录历史
        self.vehicle_states.append(vehicle_state)
        self.decision_history.append(decision)
        
        return decision
    
    def get_path_statistics(self) -> Dict:
        """获取路径统计信息"""
        if not self.path_loader.path_points:
            return {}
        
        # 计算路径长度
        total_length = self.path_loader.get_total_length()
        
        # 计算路径转向变化
        turning_angles = []
        for i in range(1, len(self.path_loader.path_points) - 1):
            p1 = self.path_loader.path_points[i-1]
            p2 = self.path_loader.path_points[i]
            p3 = self.path_loader.path_points[i+1]
            
            # 计算转向角
            angle1 = math.atan2(p2.y - p1.y, p2.x - p1.x)
            angle2 = math.atan2(p3.y - p2.y, p3.x - p2.x)
            turn_angle = abs(angle2 - angle1)
            turning_angles.append(turn_angle)
        
        avg_turn_angle = np.mean(turning_angles) if turning_angles else 0
        
        return {
            'total_length': total_length,
            'num_points': len(self.path_loader.path_points),
            'average_turn_angle_degrees': math.degrees(avg_turn_angle),
            'max_turn_angle_degrees': math.degrees(max(turning_angles)) if turning_angles else 0
        }

def create_sample_path() -> GlobalPathLoader:
    """创建示例路径"""
    # 创建一个包含直行、左转、右转的路径
    x_coords = []
    y_coords = []
    
    # 第一段：直行 (0,0) 到 (20,0)
    for i in range(21):
        x_coords.append(i)
        y_coords.append(0)
    
    # 第二段：左转圆弧 (20,0) 到 (20,10)
    center_x, center_y = 20, 0
    radius = 10
    for i in range(1, 16):  # 90度圆弧
        angle = i * math.pi / 30  # 每6度一个点
        x_coords.append(center_x + radius * math.sin(angle))
        y_coords.append(center_y + radius * math.cos(angle))
    
    # 第三段：直行 (20,10) 到 (40,10)
    for i in range(21):
        x_coords.append(20 + i)
        y_coords.append(10)
    
    # 第四段：右转圆弧 (40,10) 到 (40,-5)
    center_x, center_y = 40, 10
    radius = 15
    for i in range(1, 21):  # 120度圆弧
        angle = i * math.pi / 18
        x_coords.append(center_x + radius * math.cos(angle))
        y_coords.append(center_y + radius * math.sin(angle))
    
    # 第五段：直行到终点
    for i in range(16):
        x_coords.append(40 + i * 0.5)
        y_coords.append(-5)
    
    return GlobalPathLoader.from_coordinates(x_coords, y_coords)

def simulate_vehicle_movement(path_loader: GlobalPathLoader, 
                            num_steps: int = 100) -> List[Dict]:
    """模拟车辆沿路径运动"""
    system = PathFollowingSystem(path_loader)
    results = []
    
    # 随机偏移量模拟端到端网络的偏差
    np.random.seed(42)
    
    for i in range(num_steps):
        # 获取当前路径点（带随机偏差）
        if i < len(path_loader.path_points):
            base_point = path_loader.path_points[i]
            # 添加随机偏差
            offset_x = np.random.normal(0, 0.5)  # 0.5米的随机偏差
            offset_y = np.random.normal(0, 0.5)
            
            x = base_point.x + offset_x
            y = base_point.y + offset_y
            velocity = 2.0 + np.random.normal(0, 0.3)  # 2m/s ± 0.3
            heading = base_point.heading_angle + np.random.normal(0, 0.1)  # 0.1弧度的方向偏差
        else:
            # 超出路径范围，保持最后位置
            x = results[-1]['x'] if results else 0
            y = results[-1]['y'] if results else 0
            velocity = 0.0
            heading = results[-1]['heading'] if results else 0
        
        # 更新系统状态
        decision = system.update_vehicle_state(x, y, velocity, heading, i * 0.1)
        
        # 记录结果
        result = {
            'step': i,
            'timestamp': i * 0.1,
            'x': x,
            'y': y,
            'velocity': velocity,
            'heading': heading,
            'decision': decision
        }
        results.append(result)
        
        # 打印关键决策
        if i % 20 == 0 or decision['action'] != '直行':
            print(f"步骤 {i:3d}: 位置({x:.1f}, {y:.1f}) | "
                  f"速度 {velocity:.1f} m/s | 决策 {decision['action']:6s} | "
                  f"转向角 {decision['steering_angle']:5.1f}° | "
                  f"置信度 {decision['confidence']:.2f}")
    
    return results

def visualize_path_following(path_loader: GlobalPathLoader, 
                           simulation_results: List[Dict],
                           save_path: str = "path_following_analysis.png"):
    """可视化路径跟随结果"""
    setup_matplotlib_for_plotting()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 路径和车辆轨迹
    path_x = [p.x for p in path_loader.path_points]
    path_y = [p.y for p in path_loader.path_points]
    
    ax1.plot(path_x, path_y, 'b-', linewidth=2, label='全局路径', alpha=0.7)
    ax1.scatter(path_x, path_y, c='blue', s=20, alpha=0.5)
    
    # 车辆轨迹
    vehicle_x = [r['x'] for r in simulation_results]
    vehicle_y = [r['y'] for r in simulation_results]
    ax1.plot(vehicle_x, vehicle_y, 'r-', linewidth=2, label='车辆轨迹', alpha=0.8)
    ax1.scatter(vehicle_x, vehicle_y, c='red', s=15, alpha=0.6)
    
    ax1.set_xlabel('X坐标 (m)')
    ax1.set_ylabel('Y坐标 (m)')
    ax1.set_title('路径跟随分析')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. 速度变化
    velocities = [r['velocity'] for r in simulation_results]
    timestamps = [r['timestamp'] for r in simulation_results]
    ax2.plot(timestamps, velocities, 'g-', linewidth=2)
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('速度 (m/s)')
    ax2.set_title('车辆速度变化')
    ax2.grid(True, alpha=0.3)
    
    # 3. 转向角度
    steering_angles = [r['decision']['steering_angle'] for r in simulation_results]
    ax3.plot(timestamps, steering_angles, 'purple', linewidth=2)
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('转向角 (度)')
    ax3.set_title('转向角度变化')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 4. 决策统计
    actions = [r['decision']['action'] for r in simulation_results]
    action_counts = {}
    for action in actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
    ax4.bar(action_counts.keys(), action_counts.values(), 
           color=colors[:len(action_counts)])
    ax4.set_xlabel('驾驶行为')
    ax4.set_ylabel('出现次数')
    ax4.set_title('驾驶行为统计')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def main():
    """主函数 - 演示路径跟随系统"""
    print("🛣️  基于全局路径点的车辆转向决策系统")
    print("=" * 60)
    
    # 1. 创建示例路径
    print("📍 创建示例全局路径...")
    path_loader = create_sample_path()
    
    # 创建路径跟随系统以获取路径统计
    path_system = PathFollowingSystem(path_loader)
    path_stats = path_system.get_path_statistics()
    print(f"✅ 路径创建完成:")
    print(f"   路径点数: {path_stats['num_points']}")
    print(f"   路径长度: {path_stats['total_length']:.1f} m")
    print(f"   平均转向角: {path_stats['average_turn_angle_degrees']:.1f}°")
    print(f"   最大转向角: {path_stats['max_turn_angle_degrees']:.1f}°")
    
    # 2. 模拟车辆运动
    print("\n🚗 模拟车辆路径跟随...")
    simulation_results = simulate_vehicle_movement(path_loader, num_steps=80)
    
    # 3. 生成可视化
    print("\n📊 生成可视化分析...")
    viz_path = visualize_path_following(path_loader, simulation_results)
    print(f"✅ 可视化图表已保存: {viz_path}")
    
    # 4. 决策结果统计
    print("\n📈 决策结果统计:")
    decisions = [r['decision'] for r in simulation_results]
    action_counts = {}
    confidence_sum = 0
    total_decisions = len(decisions)
    
    for decision in decisions:
        action = decision['action']
        action_counts[action] = action_counts.get(action, 0) + 1
        confidence_sum += decision['confidence']
    
    avg_confidence = confidence_sum / total_decisions if total_decisions > 0 else 0
    
    for action, count in sorted(action_counts.items()):
        percentage = count / total_decisions * 100
        print(f"   {action}: {count} 次 ({percentage:.1f}%)")
    
    print(f"\n   平均决策置信度: {avg_confidence:.2f}")
    
    # 5. 保存详细结果
    print("\n💾 保存详细结果...")
    
    # 转换为DataFrame并保存
    df_results = pd.DataFrame([
        {
            'step': r['step'],
            'timestamp': r['timestamp'],
            'x': r['x'],
            'y': r['y'],
            'velocity': r['velocity'],
            'heading_degrees': math.degrees(r['heading']),
            'action': r['decision']['action'],
            'steering_angle_degrees': r['decision']['steering_angle'],
            'confidence': r['decision']['confidence'],
            'recommended_speed': r['decision']['recommended_speed'],
            'nearest_distance': r['decision']['nearest_distance'],
            'heading_error_degrees': r['decision']['heading_error']
        }
        for r in simulation_results
    ])
    
    df_results.to_csv('path_following_results.csv', index=False)
    print("✅ 详细结果已保存: path_following_results.csv")
    
    # 6. 显示关键决策示例
    print("\n🎯 关键决策示例:")
    print("-" * 80)
    
    # 显示非直行决策
    interesting_decisions = [r for r in simulation_results if r['decision']['action'] != '直行']
    
    for i, decision_data in enumerate(interesting_decisions[:10]):  # 显示前10个非直行决策
        r = decision_data
        d = r['decision']
        print(f"{i+1:2d}. 步骤 {r['step']:3d} | 位置({r['x']:5.1f}, {r['y']:5.1f}) | "
              f"速度 {r['velocity']:4.1f} m/s | "
              f"行为 {d['action']:8s} | 转向角 {d['steering_angle']:5.1f}° | "
              f"置信度 {d['confidence']:.2f}")
    
    print(f"\n🎉 路径跟随系统演示完成！")
    print(f"📁 输出文件:")
    print(f"   - {viz_path}")
    print(f"   - path_following_results.csv")

if __name__ == "__main__":
    main()