#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹转向决策系统使用示例
演示如何处理实际CSV数据的完整流程

作者：MiniMax Agent
"""

import pandas as pd
import numpy as np
from trajectory_classifier import (
    TrajectoryTurnClassifier, 
    TrajectoryMatcher, 
    TrajectoryDecisionMaker,
    load_and_analyze_trajectory
)

def create_sample_csv_data():
    """创建示例CSV数据文件"""
    print("📊 创建示例CSV数据...")
    
    # 生成400个轨迹点的示例数据
    np.random.seed(42)  # 确保结果可重现
    
    # 时间序列
    timestamps = np.arange(0, 40, 0.1)  # 40秒，每0.1秒一个点
    
    # 生成轨迹：起步 -> 直行 -> 左转 -> 直行 -> 右转 -> 停车
    segments = [
        (0, 40, "起步"),      # 0-4秒: 起步
        (40, 120, "直行"),    # 4-12秒: 直行
        (120, 180, "左转"),   # 12-18秒: 左转
        (180, 280, "直行"),   # 18-28秒: 直行
        (280, 340, "右转"),   # 28-34秒: 右转
        (340, 400, "停车")    # 34-40秒: 停车
    ]
    
    data_points = []
    x, y, z = 0, 0, 0
    
    for start_idx, end_idx, action in segments:
        start_time = start_idx * 0.1
        end_time = end_idx * 0.1
        
        for i in range(start_idx, end_idx):
            timestamp = i * 0.1
            
            if action == "起步":
                velocity = 0.5 + (i - start_idx) * 0.1
                steering_angle = 0
                x += velocity * 0.1
            elif action == "直行":
                velocity = 3.0 + np.random.normal(0, 0.2)
                steering_angle = np.random.normal(0, 0.1)
                x += velocity * 0.1
            elif action == "左转":
                velocity = 2.0 + np.random.normal(0, 0.3)
                steering_angle = 0.5 + np.random.normal(0, 0.2)
                # 模拟左转弧线
                x += velocity * 0.1 * np.cos(steering_angle)
                y += velocity * 0.1 * np.sin(steering_angle)
            elif action == "右转":
                velocity = 2.0 + np.random.normal(0, 0.3)
                steering_angle = -0.5 + np.random.normal(0, 0.2)
                # 模拟右转弧线
                x += velocity * 0.1 * np.cos(steering_angle)
                y += velocity * 0.1 * np.sin(steering_angle)
            elif action == "停车":
                velocity = max(0, 2.0 - (i - start_idx) * 0.2)
                steering_angle = np.random.normal(0, 0.1)
                x += velocity * 0.1
            
            # 确保速度不为负
            velocity = max(0, velocity)
            
            data_points.append({
                'timestamp': timestamp,
                'x': x,
                'y': y,
                'z': z,
                'velocity': velocity,
                'steering_angle': steering_angle
            })
    
    # 创建DataFrame
    df = pd.DataFrame(data_points)
    
    # 保存到CSV文件
    df.to_csv('sample_vehicle_trajectory.csv', index=False)
    print(f"✅ 示例数据已保存到 sample_vehicle_trajectory.csv ({len(df)} 个数据点)")
    
    return df

def analyze_trajectory_file(csv_file_path):
    """分析轨迹文件的完整流程"""
    print(f"\n🔍 分析轨迹文件: {csv_file_path}")
    print("=" * 50)
    
    try:
        # 使用主函数分析轨迹
        result = load_and_analyze_trajectory(csv_file_path)
        
        # 显示分析结果
        print_analysis_result(result)
        
        # 保存分类结果
        output_file = csv_file_path.replace('.csv', '_classified.csv')
        result['classified_data'].to_csv(output_file, index=False)
        print(f"✅ 分类结果已保存到: {output_file}")
        
        return result
        
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        return None

def print_analysis_result(result):
    """打印分析结果"""
    print("\n📊 分析结果:")
    print("-" * 30)
    
    # 当前状态
    current = result['current_state']
    print(f"🚗 当前状态:")
    print(f"   行为: {current['action']}")
    print(f"   速度: {current['velocity']:.2f} m/s")
    print(f"   转角: {current['steering_angle']:.3f} rad")
    print(f"   位置: ({current['position'][0]:.2f}, {current['position'][1]:.2f}, {current['position'][2]:.2f})")
    
    # 轨迹匹配
    match = result['trajectory_match']
    print(f"\n🎯 轨迹匹配:")
    print(f"   相似度: {match['similarity']:.3f}")
    print(f"   置信度: {match['confidence']}")
    if match['matched_position'] is not None:
        pos = match['matched_position']
        print(f"   匹配位置: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    # 预测和建议
    print(f"\n🔮 预测和建议:")
    print(f"   预测下一步: {result['next_action_prediction']}")
    print(f"   决策建议: {result['decision_advice']['recommendation']}")
    
    if result['decision_advice']['attention_points']:
        print(f"   注意要点: {', '.join(result['decision_advice']['attention_points'])}")
    
    # 分类统计
    classified_data = result['classified_data']
    action_counts = classified_data['action'].value_counts()
    print(f"\n📈 行为统计:")
    for action, count in action_counts.items():
        percentage = count / len(classified_data) * 100
        print(f"   {action}: {count} 次 ({percentage:.1f}%)")

def compare_with_historical_data(historical_file, current_file):
    """对比历史轨迹和当前轨迹"""
    print(f"\n🔄 对比历史轨迹和当前轨迹")
    print("=" * 50)
    
    try:
        # 加载数据
        historical_data = pd.read_csv(historical_file)
        current_data = pd.read_csv(current_file)
        
        print(f"历史轨迹: {len(historical_data)} 个数据点")
        print(f"当前轨迹: {len(current_data)} 个数据点")
        
        # 创建分析器
        classifier = TrajectoryTurnClassifier(
            velocity_threshold=0.5,
            angle_threshold=0.3
        )
        matcher = TrajectoryMatcher()
        decision_maker = TrajectoryDecisionMaker(classifier, matcher)
        
        # 进行对比分析
        result = decision_maker.analyze_current_state(historical_data, current_data)
        
        print("\n📊 对比分析结果:")
        print_analysis_result(result)
        
        # 比较分类结果
        historical_classified = classifier.classify_trajectory(historical_data)
        current_classified = classifier.classify_trajectory(current_data)
        
        print(f"\n📈 历史轨迹行为分布:")
        for action, count in historical_classified['action'].value_counts().items():
            percentage = count / len(historical_classified) * 100
            print(f"   {action}: {percentage:.1f}%")
        
        print(f"\n📈 当前轨迹行为分布:")
        for action, count in current_classified['action'].value_counts().items():
            percentage = count / len(current_classified) * 100
            print(f"   {action}: {percentage:.1f}%")
        
        return result
        
    except Exception as e:
        print(f"❌ 对比分析失败: {str(e)}")
        return None

def main():
    """主函数 - 演示完整的使用流程"""
    print("🚗 轨迹转向决策系统使用示例")
    print("=" * 60)
    
    # 1. 创建示例数据
    sample_data = create_sample_csv_data()
    
    # 2. 演示基本使用
    print("\n" + "="*60)
    print("🔹 演示1: 基本轨迹分析")
    print("="*60)
    result1 = analyze_trajectory_file('sample_vehicle_trajectory.csv')
    
    # 3. 演示历史数据对比
    print("\n" + "="*60)
    print("🔹 演示2: 历史数据对比分析")
    print("="*60)
    
    # 创建历史数据（前200个点）和当前数据（后200个点）
    historical_data = sample_data.iloc[:200].copy()
    current_data = sample_data.iloc[200:].copy()
    
    historical_data.to_csv('historical_trajectory.csv', index=False)
    current_data.to_csv('current_trajectory.csv', index=False)
    
    print("📁 已生成历史轨迹和当前轨迹文件")
    
    result2 = compare_with_historical_data('historical_trajectory.csv', 'current_trajectory.csv')
    
    # 4. 演示自定义参数
    print("\n" + "="*60)
    print("🔹 演示3: 自定义参数分析")
    print("="*60)
    
    try:
        # 使用自定义参数
        custom_classifier = TrajectoryTurnClassifier(
            velocity_threshold=1.0,  # 提高停车阈值
            angle_threshold=0.5,     # 提高转向阈值
            angle_window=5          # 增大角度计算窗口
        )
        
        custom_data = custom_classifier.classify_trajectory(sample_data)
        
        print("📊 自定义参数分类结果:")
        action_counts = custom_data['action'].value_counts()
        for action, count in action_counts.items():
            percentage = count / len(custom_data) * 100
            print(f"   {action}: {count} 次 ({percentage:.1f}%)")
        
        # 保存自定义结果
        custom_data.to_csv('custom_classified_trajectory.csv', index=False)
        print("✅ 自定义分类结果已保存到: custom_classified_trajectory.csv")
        
    except Exception as e:
        print(f"❌ 自定义参数分析失败: {str(e)}")
    
    # 5. 总结
    print("\n" + "="*60)
    print("🎉 使用示例完成！")
    print("="*60)
    
    print("\n📁 生成的文件:")
    print("- sample_vehicle_trajectory.csv: 原始示例数据")
    print("- sample_vehicle_trajectory_classified.csv: 基本分析结果")
    print("- historical_trajectory.csv: 历史轨迹数据")
    print("- current_trajectory.csv: 当前轨迹数据")
    print("- custom_classified_trajectory.csv: 自定义参数分析结果")
    
    print("\n💡 使用提示:")
    print("1. 将您的实际CSV数据替换示例文件")
    print("2. 根据车辆特性调整分类参数")
    print("3. 使用历史轨迹数据进行位置推断")
    print("4. 结合决策建议进行实际应用")

if __name__ == "__main__":
    main()