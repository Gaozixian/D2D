#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径跟随决策系统完整演示
非交互式版本

作者：MiniMax Agent
"""

import numpy as np
import pandas as pd
import math
from simple_path_following import SimplePathFollower, create_sample_path_csv

def main():
    """主函数 - 完整演示"""
    print("🛣️  基于全局路径点的车辆转向决策系统")
    print("=" * 60)
    
    # 1. 创建和加载路径
    print("📍 准备路径数据...")
    path_data = create_sample_path_csv()
    follower = SimplePathFollower.from_csv('sample_global_path.csv', max_velocity_kmh=15.0)
    
    print(f"✅ 路径加载完成:")
    print(f"   路径点数: {len(follower.path_points)}")
    print(f"   路径范围: X({follower.path_points[:, 0].min():.1f}, {follower.path_points[:, 0].max():.1f})")
    print(f"             Y({follower.path_points[:, 1].min():.1f}, {follower.path_points[:, 1].max():.1f})")
    
    # 2. 批量测试不同场景
    print(f"\n🚗 测试不同驾驶场景...")
    
    test_scenarios = [
        # (x, y, velocity, heading, description)
        (5.0, 0.0, 2.0, 0.0, "直线行驶"),
        (15.0, 1.0, 1.8, 0.2, "轻微偏离路径"),
        (35.0, 5.0, 1.5, 0.8, "准备左转"),
        (42.0, 8.0, 1.2, 1.0, "左转进行中"),
        (50.0, 15.0, 2.0, 0.0, "左转完成直行"),
        (70.0, 15.0, 2.2, 0.0, "直线行驶"),
        (85.0, 20.0, 1.5, 0.5, "准备右转"),
        (95.0, 10.0, 1.0, -0.5, "右转进行中"),
        (105.0, 5.0, 1.8, 0.0, "右转完成"),
        (120.0, 0.0, 0.0, 0.0, "停车状态"),
    ]
    
    results = []
    
    for i, (x, y, velocity, heading, desc) in enumerate(test_scenarios):
        decision = follower.make_decision(x, y, velocity, heading)
        
        result = {
            'scenario': desc,
            'x': x,
            'y': y,
            'velocity': velocity,
            'heading_degrees': math.degrees(heading),
            'action': decision['action'],
            'steering_angle_degrees': decision['steering_angle_degrees'],
            'recommended_speed_ms': decision['recommended_speed_ms'],
            'distance_to_path': decision['distance_to_path_m'],
            'confidence': decision['confidence']
        }
        results.append(result)
        
        print(f"\n场景 {i+1}: {desc}")
        print(f"  位置: ({x:.1f}, {y:.1f}) | 速度: {velocity:.1f} m/s | 朝向: {math.degrees(heading):.1f}°")
        print(f"  决策: {decision['action']} | 转向角: {decision['steering_angle_degrees']:5.1f}°")
        print(f"  推荐速度: {decision['recommended_speed_ms']:.2f} m/s | 路径距离: {decision['distance_to_path_m']:.2f} m")
        print(f"  置信度: {decision['confidence']:.2f}")
    
    # 3. 统计分析
    print(f"\n📊 决策统计分析:")
    print("-" * 40)
    
    action_counts = {}
    for result in results:
        action = result['action']
        action_counts[action] = action_counts.get(action, 0) + 1
    
    for action, count in sorted(action_counts.items()):
        percentage = count / len(results) * 100
        print(f"  {action}: {count} 次 ({percentage:.1f}%)")
    
    avg_confidence = np.mean([r['confidence'] for r in results])
    print(f"\n  平均置信度: {avg_confidence:.2f}")
    
    # 4. 保存结果
    print(f"\n💾 保存测试结果...")
    df_results = pd.DataFrame(results)
    df_results.to_csv('path_following_test_results.csv', index=False)
    print("✅ 结果已保存到: path_following_test_results.csv")
    
    # 5. 参数调优建议
    print(f"\n⚙️  参数调优建议:")
    print("-" * 30)
    print(f"当前设置:")
    print(f"  最大速度: {follower.max_velocity_ms * 3.1:.1f} km/h")
    print(f"  直行阈值: {math.degrees(follower.straight_threshold):.1f}°")
    print(f"  小转向阈值: {math.degrees(follower.small_turn_threshold):.1f}°")
    print(f"  大转向阈值: {math.degrees(follower.large_turn_threshold):.1f}°")
    print(f"  搜索半径: {follower.search_radius:.1f} m")
    
    print(f"\n调整建议:")
    print(f"  - 如果经常误判转向: 增大转向阈值")
    print(f"  - 如果对偏差过于敏感: 增大搜索半径")
    print(f"  - 如果速度控制不合理: 调整速度比例系数")
    
    # 6. 使用指南
    print(f"\n📖 实际使用指南:")
    print("-" * 30)
    print(f"1. 准备路径数据:")
    print(f"   - CSV文件包含 x, y 列")
    print(f"   - 路径点按行驶顺序排列")
    print(f"   - 建议点间距 1-5 米")
    
    print(f"\n2. 初始化系统:")
    print(f"   follower = SimplePathFollower.from_csv('your_path.csv')")
    
    print(f"\n3. 实时决策:")
    print(f"   decision = follower.make_decision(x, y, velocity, heading)")
    
    print(f"\n4. 获取结果:")
    print(f"   action = decision['action']")
    print(f"   steering_angle = decision['steering_angle_degrees']")
    print(f"   recommended_speed = decision['recommended_speed_ms']")
    
    print(f"\n🎉 演示完成！")
    print(f"📁 生成的文件:")
    print(f"   - sample_global_path.csv: 示例路径数据")
    print(f"   - path_following_test_results.csv: 测试结果")

if __name__ == "__main__":
    main()