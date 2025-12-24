#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版路径跟随决策系统 - 专门用于处理用户实际数据
针对15km/h最高速度，端到端网络的路径偏差问题

作者：MiniMax Agent
使用方法：
1. 准备路径点CSV文件 (包含x, y列)
2. 实时输入车辆位置坐标
3. 获取转向决策结果
"""

import numpy as np
import pandas as pd
import math
from typing import List, Tuple, Dict, Optional

class SimplePathFollower:
    """简化版路径跟随器"""
    
    def __init__(self, path_points: List[Tuple[float, float]], 
                 max_velocity_kmh: float = 15.0):
        """
        初始化路径跟随器
        
        Args:
            path_points: 路径点列表 [(x1, y1), (x2, y2), ...]
            max_velocity_kmh: 最大速度 (km/h)
        """
        self.path_points = np.array(path_points)
        self.max_velocity_ms = max_velocity_kmh / 3.6  # 转换为m/s
        self.last_matched_index = -1
        
        # 转向阈值 (弧度)
        self.straight_threshold = math.radians(2.0)    # 直行阈值
        self.small_turn_threshold = math.radians(10.0)  # 小转向阈值
        self.large_turn_threshold = math.radians(25.0)  # 大转向阈值
        self.stop_threshold = 0.5  # 停车阈值 (m/s)
        
        # 搜索半径
        self.search_radius = 5.0  # 米
        
    @classmethod
    def from_csv(cls, csv_file: str, max_velocity_kmh: float = 15.0) -> 'SimplePathFollower':
        """从CSV文件加载路径点"""
        df = pd.read_csv(csv_file)
        
        if 'x' not in df.columns or 'y' not in df.columns:
            raise ValueError("CSV文件必须包含x和y列")
        
        path_points = list(zip(df['x'].values, df['y'].values))
        return cls(path_points, max_velocity_kmh)
    
    def find_nearest_path_point(self, vehicle_x: float, vehicle_y: float) -> Tuple[int, float]:
        """
        找到距离车辆最近的路径点
        
        Returns:
            Tuple[int, float]: (路径点索引, 距离)
        """
        # 计算所有路径点的距离
        distances = np.sqrt((self.path_points[:, 0] - vehicle_x)**2 + 
                           (self.path_points[:, 1] - vehicle_y)**2)
        
        # 找到最近的点
        nearest_index = np.argmin(distances)
        min_distance = distances[nearest_index]
        
        # 如果距离太远，扩大搜索范围
        if min_distance > self.search_radius:
            # 在全路径中搜索
            nearest_index = np.argmin(distances)
            min_distance = distances[nearest_index]
        
        self.last_matched_index = nearest_index
        return nearest_index, min_distance
    
    def calculate_steering_angle(self, vehicle_x: float, vehicle_y: float, 
                               vehicle_heading: float, lookahead_distance: float = 8.0) -> float:
        """
        计算转向角
        
        Args:
            vehicle_x, vehicle_y: 车辆位置
            vehicle_heading: 车辆朝向 (弧度)
            lookahead_distance: 前瞻距离 (米)
            
        Returns:
            float: 转向角 (弧度)
        """
        # 找到最近路径点
        nearest_index, _ = self.find_nearest_path_point(vehicle_x, vehicle_y)
        
        # 找到前瞻点
        lookahead_point = self._find_lookahead_point(nearest_index, lookahead_distance)
        
        if lookahead_point is None:
            return 0.0
        
        # 计算目标朝向
        target_heading = math.atan2(
            lookahead_point[1] - vehicle_y,
            lookahead_point[0] - vehicle_x
        )
        
        # 计算角度差
        angle_diff = target_heading - vehicle_heading
        
        # 处理角度跳跃
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        return angle_diff
    
    def _find_lookahead_point(self, start_index: int, target_distance: float) -> Optional[Tuple[float, float]]:
        """找到前瞻路径点"""
        if start_index >= len(self.path_points) - 1:
            return self.path_points[-1]
        
        accumulated_distance = 0.0
        
        for i in range(start_index, len(self.path_points) - 1):
            p1 = self.path_points[i]
            p2 = self.path_points[i + 1]
            
            segment_distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            accumulated_distance += segment_distance
            
            if accumulated_distance >= target_distance:
                return p2
        
        return self.path_points[-1]
    
    def make_decision(self, vehicle_x: float, vehicle_y: float, 
                     vehicle_velocity: float = 0.0, 
                     vehicle_heading: float = 0.0) -> Dict:
        """
        做出转向决策
        
        Args:
            vehicle_x, vehicle_y: 车辆位置
            vehicle_velocity: 车辆速度 (m/s)
            vehicle_heading: 车辆朝向 (弧度)
            
        Returns:
            Dict: 决策结果
        """
        # 1. 计算转向角
        steering_angle = self.calculate_steering_angle(vehicle_x, vehicle_y, vehicle_heading)
        
        # 2. 找到最近路径点
        nearest_index, distance_to_path = self.find_nearest_path_point(vehicle_x, vehicle_y)
        
        # 3. 判断行为
        action = self._classify_action(vehicle_velocity, steering_angle, distance_to_path)
        
        # 4. 计算推荐速度
        recommended_speed = self._calculate_recommended_speed(
            vehicle_velocity, steering_angle, distance_to_path
        )
        
        # 5. 计算置信度
        confidence = self._calculate_confidence(distance_to_path, steering_angle)
        
        return {
            'action': action,
            'steering_angle_degrees': math.degrees(steering_angle),
            'recommended_speed_ms': recommended_speed,
            'distance_to_path_m': distance_to_path,
            'nearest_path_index': nearest_index,
            'confidence': confidence,
            'vehicle_heading_degrees': math.degrees(vehicle_heading),
            'max_speed_kmh': self.max_velocity_ms * 3.6
        }
    
    def _classify_action(self, velocity: float, steering_angle: float, distance: float) -> str:
        """分类驾驶行为"""
        abs_angle = abs(steering_angle)
        
        # 速度判断
        if velocity < self.stop_threshold:
            return '停车'
        
        # 距离判断 - 离路径太远
        if distance > 3.0:
            return '偏离路径'
        
        # 转向角度判断
        if abs_angle < self.straight_threshold:
            return '直行'
        elif abs_angle < self.small_turn_threshold:
            return '小转向'
        elif abs_angle < self.large_turn_threshold:
            return '转向'
        else:
            return '急转弯'
    
    def _calculate_recommended_speed(self, current_velocity: float, 
                                   steering_angle: float, distance: float) -> float:
        """计算推荐速度"""
        base_speed = self.max_velocity_ms
        
        # 根据转向角度调整
        abs_angle = abs(steering_angle)
        if abs_angle > self.large_turn_threshold:
            base_speed *= 0.3  # 急转弯
        elif abs_angle > self.small_turn_threshold:
            base_speed *= 0.6  # 转向
        elif abs_angle > self.straight_threshold:
            base_speed *= 0.8  # 小转向
        
        # 根据路径偏差调整
        if distance > 2.0:
            base_speed *= 0.7  # 偏离路径减速
        elif distance > 1.0:
            base_speed *= 0.85
        
        return min(base_speed, self.max_velocity_ms)
    
    def _calculate_confidence(self, distance: float, steering_angle: float) -> float:
        """计算决策置信度"""
        confidence = 1.0
        
        # 距离置信度
        if distance < 0.5:
            confidence *= 1.0
        elif distance < 2.0:
            confidence *= 0.8
        elif distance < 5.0:
            confidence *= 0.6
        else:
            confidence *= 0.3
        
        # 转向角度置信度
        abs_angle = abs(steering_angle)
        if abs_angle > math.radians(45):
            confidence *= 0.5  # 过大转向可能不可信
        
        return max(0.0, min(1.0, confidence))

def create_sample_path_csv():
    """创建示例路径CSV文件"""
    # 生成一个包含直行和转弯的路径
    x_coords = []
    y_coords = []
    
    # 直行段
    for i in range(20):
        x_coords.append(i * 2.0)
        y_coords.append(0.0)
    
    # 左转圆弧
    center_x, center_y = 40.0, 0.0
    radius = 15.0
    for i in range(1, 16):
        angle = i * math.pi / 30  # 90度圆弧
        x_coords.append(center_x + radius * math.sin(angle))
        y_coords.append(center_y + radius * math.cos(angle))
    
    # 直行段
    for i in range(25):
        x_coords.append(40.0 + i * 2.0)
        y_coords.append(15.0)
    
    # 右转圆弧
    center_x, center_y = 90.0, 15.0
    radius = 20.0
    for i in range(1, 21):
        angle = i * math.pi / 18  # 120度圆弧
        x_coords.append(center_x + radius * math.cos(angle))
        y_coords.append(center_y + radius * math.sin(angle))
    
    # 保存到CSV
    df = pd.DataFrame({'x': x_coords, 'y': y_coords})
    df.to_csv('sample_global_path.csv', index=False)
    
    return df

def demo_usage():
    """演示使用方法"""
    print("🛣️  简化版路径跟随系统演示")
    print("=" * 50)
    
    # 1. 创建示例路径
    print("📍 创建示例路径...")
    path_data = create_sample_path_csv()
    print(f"✅ 路径已保存到 sample_global_path.csv ({len(path_data)} 个点)")
    
    # 2. 初始化路径跟随器
    print("\n🔧 初始化路径跟随器...")
    follower = SimplePathFollower.from_csv('sample_global_path.csv', max_velocity_kmh=15.0)
    print("✅ 路径跟随器初始化完成")
    
    # 3. 模拟车辆位置输入
    print("\n🚗 模拟车辆位置输入...")
    
    # 模拟几个不同位置的决策
    test_positions = [
        (5.0, 0.5, 2.0, 0.1),    # 直行段
        (25.0, 2.0, 1.8, 0.3),   # 接近转弯
        (45.0, 8.0, 1.5, 0.8),   # 转弯中
        (65.0, 15.0, 2.0, 0.0),  # 转弯后直行
        (95.0, 5.0, 1.2, -0.5),  # 右转段
    ]
    
    for i, (x, y, velocity, heading) in enumerate(test_positions):
        decision = follower.make_decision(x, y, velocity, heading)
        
        print(f"\n测试点 {i+1}:")
        print(f"  位置: ({x:.1f}, {y:.1f})")
        print(f"  速度: {velocity:.1f} m/s")
        print(f"  朝向: {math.degrees(heading):.1f}°")
        print(f"  决策: {decision['action']}")
        print(f"  转向角: {decision['steering_angle_degrees']:.1f}°")
        print(f"  推荐速度: {decision['recommended_speed_ms']:.1f} m/s")
        print(f"  路径距离: {decision['distance_to_path_m']:.2f} m")
        print(f"  置信度: {decision['confidence']:.2f}")
    
    print(f"\n🎉 演示完成！")

def analyze_csv_file(csv_file: str):
    """分析用户提供的路径CSV文件"""
    print(f"📂 分析路径文件: {csv_file}")
    print("=" * 50)
    
    try:
        # 加载路径
        follower = SimplePathFollower.from_csv(csv_file, max_velocity_kmh=15.0)
        
        # 显示路径信息
        path_points = follower.path_points
        print(f"✅ 成功加载路径:")
        print(f"   路径点数: {len(path_points)}")
        print(f"   X范围: {path_points[:, 0].min():.1f} ~ {path_points[:, 0].max():.1f}")
        print(f"   Y范围: {path_points[:, 1].min():.1f} ~ {path_points[:, 1].max():.1f}")
        
        # 计算路径长度
        total_length = 0.0
        for i in range(len(path_points) - 1):
            dx = path_points[i+1, 0] - path_points[i, 0]
            dy = path_points[i+1, 1] - path_points[i, 1]
            total_length += math.sqrt(dx*dx + dy*dy)
        
        print(f"   路径总长: {total_length:.1f} m")
        print(f"   最大速度: {follower.max_velocity_ms * 3.6:.1f} km/h")
        
        print(f"\n💡 使用提示:")
        print(f"   - 路径点数量: {len(path_points)} (建议 > 20 个点)")
        print(f"   - 可以通过修改参数调整转向灵敏度")
        print(f"   - 系统会自动处理路径偏差和端到端网络的不准确性")
        
        return follower
        
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        return None

def interactive_test(follower: SimplePathFollower):
    """交互式测试"""
    print(f"\n🔄 交互式测试 (输入 'quit' 退出)")
    print("-" * 50)
    print("请输入车辆状态: x y velocity heading")
    print("示例: 10.5 2.3 2.0 0.1")
    
    while True:
        try:
            user_input = input("\n输入状态 (x y velocity heading): ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            values = user_input.split()
            if len(values) != 4:
                print("❌ 请输入4个数值: x y velocity heading")
                continue
            
            x, y, velocity, heading = map(float, values)
            
            decision = follower.make_decision(x, y, velocity, heading)
            
            print(f"\n📊 决策结果:")
            print(f"   行为: {decision['action']}")
            print(f"   转向角: {decision['steering_angle_degrees']:.1f}°")
            print(f"   推荐速度: {decision['recommended_speed_ms']:.2f} m/s")
            print(f"   路径距离: {decision['distance_to_path_m']:.2f} m")
            print(f"   置信度: {decision['confidence']:.2f}")
            print(f"   最近路径点: {decision['nearest_path_index']}")
        
        except ValueError:
            print("❌ 请输入有效的数值")
        except KeyboardInterrupt:
            break
    
    print("👋 交互式测试结束")

if __name__ == "__main__":
    # 演示基本使用
    demo_usage()
    
    print(f"\n" + "="*60)
    print("📁 使用您的路径文件")
    print("="*60)
    
    # 提示用户使用自己的文件
    csv_file = input("请输入您的路径CSV文件路径 (或回车使用示例): ").strip()
    
    if csv_file:
        follower = analyze_csv_file(csv_file)
        if follower:
            interactive_test(follower)
    else:
        print("💡 您可以将自己的路径点数据保存为CSV文件，包含x和y列")
        print("   然后修改csv_file变量来使用您的数据")