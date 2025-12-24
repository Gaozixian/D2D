#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹转向分类器 - 实用版本
用于处理实际CSV数据的轨迹分析和转向决策

作者：MiniMax Agent
使用说明：
1. 确保CSV文件包含列：timestamp, x, y, z, velocity, steering_angle
2. 调用TrajectoryTurnClassifier进行分类
3. 使用TrajectoryMatcher进行轨迹匹配
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from typing import Dict, List, Tuple, Optional
import warnings

class TrajectoryTurnClassifier:
    """轨迹转向分类器"""
    
    def __init__(self, velocity_threshold: float = 0.5, 
                 angle_threshold: float = 0.3,
                 angle_window: int = 3):
        """
        初始化分类器
        
        Args:
            velocity_threshold: 速度阈值 (m/s)，低于此值认为是停车
            angle_threshold: 转角阈值 (rad)，大于此值认为是转向
            angle_window: 计算角度变化的窗口大小
        """
        self.velocity_threshold = velocity_threshold
        self.angle_threshold = angle_threshold
        self.angle_window = angle_window
    
    def classify_trajectory(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        对轨迹数据进行转向分类
        
        Args:
            data: 包含轨迹数据的DataFrame，必须包含列：
                 ['timestamp', 'x', 'y', 'z', 'velocity', 'steering_angle']
                 
        Returns:
            DataFrame: 包含分类结果的数据，添加了'action'列
        """
        # 验证输入数据
        required_columns = ['timestamp', 'x', 'y', 'z', 'velocity', 'steering_angle']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"数据必须包含列: {required_columns}")
        
        result_data = data.copy()
        classifications = []
        
        print(f"🔄 开始分类 {len(data)} 个轨迹点...")
        
        for i in range(len(data)):
            if i % 50 == 0:
                print(f"   进度: {i}/{len(data)} ({i/len(data)*100:.1f}%)")
            
            classification = self._classify_single_point(data, i)
            classifications.append(classification)
        
        result_data['action'] = classifications
        print(f"✅ 分类完成!")
        
        return result_data
    
    def _classify_single_point(self, data: pd.DataFrame, index: int) -> str:
        """分类单个轨迹点"""
        if index < self.angle_window:
            return "起步"  # 前几个点默认是起步
        
        current_velocity = data.iloc[index]['velocity']
        
        # 如果速度很低，认为是停车
        if current_velocity < self.velocity_threshold:
            return "停车"
        
        # 计算角度变化和速度变化
        angle_change = self._calculate_angle_change(data, index)
        velocity_change = self._calculate_velocity_change(data, index)
        
        # 起步检测：速度从低到高
        if velocity_change > 0.5 and current_velocity < 1.0:
            return "起步"
        
        # 转向检测：角度变化超过阈值
        if abs(angle_change) > self.angle_threshold:
            if angle_change > 0:
                return "左转"
            else:
                return "右转"
        
        # 默认是直行
        return "直行"
    
    def _calculate_angle_change(self, data: pd.DataFrame, index: int) -> float:
        """计算角度变化"""
        if index < self.angle_window:
            return 0
        
        current_angle = data.iloc[index]['steering_angle']
        prev_angle = data.iloc[index - self.angle_window]['steering_angle']
        
        # 处理角度跳跃（-π 到 π）
        angle_diff = current_angle - prev_angle
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        return angle_diff
    
    def _calculate_velocity_change(self, data: pd.DataFrame, index: int) -> float:
        """计算速度变化"""
        if index < self.angle_window:
            return 0
        
        current_velocity = data.iloc[index]['velocity']
        prev_velocity = data.iloc[index - self.angle_window]['velocity']
        
        return current_velocity - prev_velocity
    
    def get_action_statistics(self, classified_data: pd.DataFrame) -> Dict[str, int]:
        """获取动作统计信息"""
        action_counts = classified_data['action'].value_counts().to_dict()
        total_points = len(classified_data)
        
        # 计算百分比
        statistics = {}
        for action, count in action_counts.items():
            percentage = count / total_points * 100
            statistics[action] = {
                'count': count,
                'percentage': percentage
            }
        
        return statistics

class TrajectoryMatcher:
    """轨迹匹配器"""
    
    def __init__(self):
        pass
    
    def match_current_position(self, historical_data: pd.DataFrame, 
                             current_data: pd.DataFrame, 
                             window_size: int = 20) -> Dict:
        """
        匹配当前轨迹与历史轨迹，确定当前位置
        
        Args:
            historical_data: 历史轨迹数据
            current_data: 当前轨迹数据
            window_size: 匹配窗口大小
            
        Returns:
            Dict: 匹配结果，包含相似度、匹配位置等信息
        """
        if len(current_data) < window_size:
            return {
                "match_score": 0,
                "matched_position": None,
                "match_index": None,
                "similarity": 0,
                "confidence": "low"
            }
        
        # 获取当前轨迹的最后一个窗口
        current_window = current_data.iloc[-window_size:][['x', 'y', 'z']].values
        
        best_match = None
        best_score = 0
        best_index = None
        
        print(f"🔍 在历史轨迹中搜索最佳匹配 (窗口大小: {window_size})...")
        
        # 在历史轨迹中寻找最佳匹配
        for i in range(len(historical_data) - window_size + 1):
            historical_window = historical_data.iloc[i:i+window_size][['x', 'y', 'z']].values
            
            # 计算相似度分数
            score = self._calculate_similarity(current_window, historical_window)
            
            if score > best_score:
                best_score = score
                best_match = i
                best_index = i + window_size - 1
        
        # 计算匹配位置
        matched_position = None
        if best_match is not None:
            matched_position = historical_data.iloc[best_index][['x', 'y', 'z']].values
        
        # 确定置信度
        confidence = "high" if best_score > 0.8 else "medium" if best_score > 0.5 else "low"
        
        result = {
            "match_score": best_score,
            "matched_position": matched_position,
            "match_index": best_index,
            "similarity": best_score,
            "confidence": confidence,
            "window_size": window_size
        }
        
        print(f"✅ 匹配完成 - 相似度: {best_score:.3f}, 置信度: {confidence}")
        
        return result
    
    def _calculate_similarity(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """计算两条轨迹的相似度"""
        if len(traj1) != len(traj2):
            return 0
        
        # 计算点对点的欧氏距离
        distances = [euclidean(traj1[i], traj2[i]) for i in range(len(traj1))]
        
        # 转换为相似度分数（距离越小，相似度越高）
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)
        
        # 综合考虑平均距离和最大距离
        similarity = 1 / (1 + avg_distance + 0.5 * max_distance)
        
        return similarity

class TrajectoryDecisionMaker:
    """轨迹决策器 - 基于轨迹分析做出转向决策"""
    
    def __init__(self, classifier: TrajectoryTurnClassifier, 
                 matcher: TrajectoryMatcher):
        self.classifier = classifier
        self.matcher = matcher
    
    def analyze_current_state(self, historical_data: pd.DataFrame, 
                            current_data: pd.DataFrame) -> Dict:
        """
        分析当前车辆状态并做出决策
        
        Args:
            historical_data: 历史轨迹数据
            current_data: 当前轨迹数据
            
        Returns:
            Dict: 包含当前状态、预测决策等信息的字典
        """
        print("🎯 开始分析当前车辆状态...")
        
        # 1. 对当前轨迹进行分类
        current_classified = self.classifier.classify_trajectory(current_data)
        
        # 2. 进行轨迹匹配
        match_result = self.matcher.match_current_position(
            historical_data, current_data, window_size=min(20, len(current_data)//2)
        )
        
        # 3. 获取当前状态
        if len(current_classified) > 0:
            current_state = self._get_current_state(current_classified)
        else:
            current_state = {
                "action": "未知",
                "velocity": 0,
                "steering_angle": 0,
                "position": None
            }
        
        # 4. 预测下一步决策
        next_action_prediction = self._predict_next_action(current_classified)
        
        # 5. 生成决策建议
        decision_advice = self._generate_decision_advice(current_state, next_action_prediction)
        
        analysis_result = {
            "current_state": current_state,
            "trajectory_match": match_result,
            "next_action_prediction": next_action_prediction,
            "decision_advice": decision_advice,
            "classified_data": current_classified
        }
        
        return analysis_result
    
    def _get_current_state(self, classified_data: pd.DataFrame) -> Dict:
        """获取当前状态"""
        latest_point = classified_data.iloc[-1]
        
        return {
            "action": latest_point['action'],
            "velocity": float(latest_point['velocity']),
            "steering_angle": float(latest_point['steering_angle']),
            "position": [float(latest_point['x']), float(latest_point['y']), float(latest_point['z'])],
            "timestamp": float(latest_point['timestamp'])
        }
    
    def _predict_next_action(self, classified_data: pd.DataFrame, window_size: int = 10) -> str:
        """预测下一个动作"""
        if len(classified_data) < window_size:
            return classified_data.iloc[-1]['action'] if len(classified_data) > 0 else "直行"
        
        # 分析最近的动作模式
        recent_data = classified_data.iloc[-window_size:]
        recent_actions = recent_data['action'].tolist()
        
        # 统计最近动作
        action_counts = {}
        for action in recent_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # 预测逻辑
        if action_counts.get('停车', 0) > window_size * 0.7:
            return "起步"
        elif action_counts.get('左转', 0) > window_size * 0.6:
            return "直行"  # 转弯后通常继续直行
        elif action_counts.get('右转', 0) > window_size * 0.6:
            return "直行"
        else:
            return "直行"  # 默认预测直行
    
    def _generate_decision_advice(self, current_state: Dict, next_prediction: str) -> Dict:
        """生成决策建议"""
        current_action = current_state['action']
        current_velocity = current_state['velocity']
        
        advice = {
            "immediate_action": current_action,
            "next_predicted_action": next_prediction,
            "recommendation": "",
            "attention_points": []
        }
        
        # 生成具体建议
        if current_action == "停车" and next_prediction == "起步":
            advice["recommendation"] = "准备起步，注意周围环境"
            advice["attention_points"].append("检查起步安全")
        elif current_action in ["左转", "右转"]:
            advice["recommendation"] = f"当前{current_action}，注意保持稳定转向"
            advice["attention_points"].append("监控转向角度")
        elif current_action == "直行":
            if current_velocity < 1.0:
                advice["recommendation"] = "当前直行且速度较低，可能需要加速"
            else:
                advice["recommendation"] = "保持直行，注意前方路况"
        
        return advice

def load_and_analyze_trajectory(csv_file_path: str, 
                               historical_data: pd.DataFrame = None) -> Dict:
    """
    加载并分析轨迹数据的主函数
    
    Args:
        csv_file_path: 当前轨迹CSV文件路径
        historical_data: 历史轨迹数据（可选）
        
    Returns:
        Dict: 完整的分析结果
    """
    print("📂 加载轨迹数据...")
    
    # 加载当前轨迹数据
    try:
        current_data = pd.read_csv(csv_file_path)
        print(f"✅ 成功加载 {len(current_data)} 个轨迹点")
    except Exception as e:
        raise Exception(f"无法加载文件 {csv_file_path}: {str(e)}")
    
    # 如果没有提供历史数据，使用当前数据作为历史数据
    if historical_data is None:
        historical_data = current_data.copy()
        print("⚠️  未提供历史数据，使用当前数据作为历史参考")
    
    # 初始化分析器
    classifier = TrajectoryTurnClassifier()
    matcher = TrajectoryMatcher()
    decision_maker = TrajectoryDecisionMaker(classifier, matcher)
    
    # 进行完整分析
    analysis_result = decision_maker.analyze_current_state(historical_data, current_data)
    
    return analysis_result

# 示例使用函数
def example_usage():
    """使用示例"""
    print("🚗 轨迹转向决策系统使用示例")
    print("=" * 40)
    
    # 1. 创建示例数据
    print("📊 创建示例数据...")
    
    # 这里可以替换为实际CSV文件路径
    example_data = pd.DataFrame({
        'timestamp': np.arange(0, 10, 0.1),
        'x': np.cumsum(np.random.randn(100) * 0.1),
        'y': np.cumsum(np.random.randn(100) * 0.1),
        'z': np.zeros(100),
        'velocity': np.abs(np.random.randn(100)) + 0.5,
        'steering_angle': np.random.randn(100) * 0.5
    })
    
    # 保存示例数据
    example_data.to_csv('example_trajectory.csv', index=False)
    print("✅ 示例数据已保存到 example_trajectory.csv")
    
    # 2. 分析轨迹
    print("\n🔍 分析轨迹...")
    result = load_and_analyze_trajectory('example_trajectory.csv')
    
    # 3. 显示结果
    print("\n📊 分析结果:")
    print(f"当前状态: {result['current_state']['action']}")
    print(f"当前速度: {result['current_state']['velocity']:.2f} m/s")
    print(f"预测下一步: {result['next_action_prediction']}")
    print(f"匹配相似度: {result['trajectory_match']['similarity']:.3f}")
    print(f"决策建议: {result['decision_advice']['recommendation']}")

if __name__ == "__main__":
    example_usage()