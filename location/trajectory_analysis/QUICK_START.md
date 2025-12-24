---
AIGC:
    ContentProducer: Minimax Agent AI
    ContentPropagator: Minimax Agent AI
    Label: AIGC
    ProduceID: "00000000000000000000000000000000"
    PropagateID: "00000000000000000000000000000000"
    ReservedCode1: 304402201c76705e39be2ee901996bfd945c30fd55b7672df88670c5468bb50b74d7e1bd022006d4e96c0ddc86c090994a93f2e39a0cc6843c0d33b5881b3710509fad3b9c4d
    ReservedCode2: 3046022100870125b38e2a5b3317bebbbb6785fb2532852b357d4e28fc9251ea147861eb73022100b2cb35c6885948c3481eaf6aeeea689056af50a3edc6fd7332bcfc7b22bbd0d7
---

# 快速开始指南

## 一分钟上手轨迹转向决策系统

### 步骤1: 准备数据
确保您的CSV文件包含以下列：
```csv
timestamp,x,y,z,velocity,steering_angle
0.0,0.0,0.0,0.0,1.2,0.0
0.1,0.12,0.0,0.0,1.3,0.0
...
```

### 步骤2: 一键分析
```python
from trajectory_classifier import load_and_analyze_trajectory

# 分析您的轨迹文件
result = load_and_analyze_trajectory('your_trajectory.csv')

# 获取结果
current_state = result['current_state']
prediction = result['next_action_prediction']
advice = result['decision_advice']['recommendation']

print(f"当前行为: {current_state['action']}")
print(f"预测下一步: {prediction}")
print(f"建议: {advice}")
```

### 步骤3: 查看输出
系统会自动生成：
- `_classified.csv`: 包含分类结果的完整数据
- 分析报告: 当前状态、预测和建议

## 核心功能

### 🎯 转向分类
- **起步**: 车辆从静止开始移动
- **直行**: 保持直线行驶
- **左转**: 向左转向
- **右转**: 向右转向  
- **停车**: 车辆停止

### 📍 位置推断
通过轨迹匹配算法，自动推断车辆在历史路径中的位置。

### 🔮 动作预测
基于当前轨迹模式，预测下一步可能的驾驶动作。

## 参数调优

根据您的车辆特性调整参数：

```python
from trajectory_classifier import TrajectoryTurnClassifier

classifier = TrajectoryTurnClassifier(
    velocity_threshold=0.5,  # 停车速度阈值 (m/s)
    angle_threshold=0.3,     # 转向角度阈值 (rad)
    angle_window=3          # 计算窗口大小
)
```

## 实际应用场景

1. **自动驾驶**: 实时驾驶行为识别
2. **驾驶分析**: 分析驾驶员习惯和行为模式
3. **路径规划**: 基于历史轨迹规划最优路径
4. **安全监控**: 异常驾驶行为检测

## 常见问题

**Q: 数据格式不对怎么办？**  
A: 确保CSV包含必需列：timestamp, x, y, z, velocity, steering_angle

**Q: 分类结果不准确？**  
A: 调整velocity_threshold和angle_threshold参数

**Q: 轨迹匹配失败？**  
A: 检查历史数据和当前数据是否使用相同坐标系

## 更多示例

运行完整示例：
```bash
python example_usage.py
```

查看详细文档：
- `README.md`: 完整技术文档
- `trajectory_classifier.py`: 核心算法实现
- `trajectory_analysis_system.py`: 可视化演示系统