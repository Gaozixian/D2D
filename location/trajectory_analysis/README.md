---
AIGC:
    ContentProducer: Minimax Agent AI
    ContentPropagator: Minimax Agent AI
    Label: AIGC
    ProduceID: "00000000000000000000000000000000"
    PropagateID: "00000000000000000000000000000000"
    ReservedCode1: 30450220363e726ad70c7b766b68ec23bd9b1dd57f255ad7b92f5f99df3b97de7f7cb67a022100f2f0272ded883320493951b1102b366428ee84f0fb540ff2396f684d8ec59338
    ReservedCode2: 3045022017f2ea1092f97bbefb39771bf47f0381c7769cdb9ef5699e4004418a8d2340a9022100c3840095a6dcdf2afeb686a917f7636c2e4ad5a4a12b361f3ce6efb826cfc6eb
---

# 轨迹转向决策系统

基于坐标位置的车辆轨迹分析和转向决策系统，能够通过历史轨迹和当前轨迹判断车辆位置并预测转向行为。

## 功能特点

- **轨迹匹配**: 通过坐标位置匹配当前轨迹与历史轨迹
- **转向分类**: 自动识别起步、直行、左转、右转、停车等行为
- **位置推断**: 基于轨迹匹配推断车辆当前位置
- **决策预测**: 预测下一步可能的转向动作
- **可视化分析**: 生成轨迹图和分类结果图表

## 系统架构

### 核心组件

1. **TrajectoryTurnClassifier**: 轨迹转向分类器
   - 基于速度、转角和位置变化进行分类
   - 可配置阈值参数
   - 支持批量处理轨迹数据

2. **TrajectoryMatcher**: 轨迹匹配器
   - 使用欧氏距离计算轨迹相似度
   - 滑动窗口匹配算法
   - 提供匹配置信度评估

3. **TrajectoryDecisionMaker**: 轨迹决策器
   - 综合分析当前状态
   - 预测下一步动作
   - 生成决策建议

## 数据格式要求

输入CSV文件必须包含以下列：

| 列名 | 类型 | 描述 | 单位 |
|------|------|------|------|
| timestamp | float | 时间戳 | 秒 |
| x | float | X坐标 | 米 |
| y | float | Y坐标 | 米 |
| z | float | Z坐标 | 米 |
| velocity | float | 速度 | m/s |
| steering_angle | float | 转角 | 弧度 |

### 示例数据格式

```csv
timestamp,x,y,z,velocity,steering_angle
0.0,0.0,0.0,0.0,1.2,0.0
0.1,0.12,0.0,0.0,1.3,0.0
0.2,0.24,0.0,0.0,1.4,0.0
...
```

## 快速开始

### 1. 基本使用

```python
from trajectory_classifier import load_and_analyze_trajectory

# 分析轨迹数据
result = load_and_analyze_trajectory('your_trajectory.csv')

# 获取当前状态
current_state = result['current_state']
print(f"当前行为: {current_state['action']}")
print(f"当前速度: {current_state['velocity']:.2f} m/s")

# 获取预测
next_action = result['next_action_prediction']
print(f"预测下一步: {next_action}")

# 获取决策建议
advice = result['decision_advice']
print(f"建议: {advice['recommendation']}")
```

### 2. 高级使用

```python
from trajectory_classifier import TrajectoryTurnClassifier, TrajectoryMatcher, TrajectoryDecisionMaker
import pandas as pd

# 加载数据
historical_data = pd.read_csv('historical_trajectory.csv')
current_data = pd.read_csv('current_trajectory.csv')

# 初始化组件
classifier = TrajectoryTurnClassifier(
    velocity_threshold=0.5,  # 速度阈值
    angle_threshold=0.3,     # 转角阈值
    angle_window=3          # 角度计算窗口
)

matcher = TrajectoryMatcher()
decision_maker = TrajectoryDecisionMaker(classifier, matcher)

# 进行分析
result = decision_maker.analyze_current_state(historical_data, current_data)
```

## 参数配置

### TrajectoryTurnClassifier 参数

- `velocity_threshold` (float): 速度阈值，低于此值认为是停车，默认 0.5 m/s
- `angle_threshold` (float): 转角阈值，大于此值认为是转向，默认 0.3 弧度
- `angle_window` (int): 计算角度变化的窗口大小，默认 3

### 分类逻辑

1. **停车**: 速度 < velocity_threshold
2. **起步**: 速度增加且速度 < 1.0 m/s
3. **左转**: 角度变化 > angle_threshold
4. **右转**: 角度变化 < -angle_threshold
5. **直行**: 其他情况

## 输出结果

### 分析结果结构

```python
{
    "current_state": {
        "action": "直行",           # 当前行为
        "velocity": 2.5,           # 当前速度 (m/s)
        "steering_angle": 0.1,     # 当前转角 (弧度)
        "position": [10.5, 20.3, 0], # 当前位置 [x, y, z]
        "timestamp": 15.2          # 时间戳
    },
    "trajectory_match": {
        "match_score": 0.85,       # 匹配分数
        "similarity": 0.85,        # 相似度
        "confidence": "high",      # 置信度: high/medium/low
        "matched_position": [10.2, 20.1, 0] # 匹配位置
    },
    "next_action_prediction": "直行", # 预测下一步动作
    "decision_advice": {
        "immediate_action": "直行",   # 当前立即动作
        "next_predicted_action": "直行", # 预测的下一步动作
        "recommendation": "保持直行，注意前方路况",
        "attention_points": ["监控前方路况"]
    },
    "classified_data": DataFrame   # 包含分类结果的完整数据
}
```

## 文件说明

### 核心文件

- `trajectory_classifier.py`: 主要的分类和决策系统
- `trajectory_analysis_system.py`: 完整的演示系统，包含数据生成和可视化
- `setup_matplotlib.py`: matplotlib配置工具

### 输出文件

- `output/historical_trajectory_classified.csv`: 历史轨迹分类结果
- `output/current_trajectory_classified.csv`: 当前轨迹分类结果
- `output/trajectory_analysis.png`: 轨迹分析可视化图
- `output/trajectory_comparison.png`: 轨迹对比图

## 使用建议

### 数据质量

1. **采样频率**: 建议采样频率 ≥ 10Hz
2. **数据完整性**: 确保时间戳连续，位置数据准确
3. **坐标系**: 确保历史数据和当前数据使用相同坐标系

### 参数调优

1. **速度阈值**: 根据车辆特性调整停车/起步阈值
2. **转角阈值**: 根据转向灵敏度调整转角阈值
3. **匹配窗口**: 根据轨迹复杂度调整匹配窗口大小

### 性能优化

1. **大数据处理**: 对于大数据集，考虑分批处理
2. **实时应用**: 可调整算法复杂度以满足实时性要求
3. **精度平衡**: 在计算精度和处理速度间找到平衡

## 扩展功能

### 可扩展的方向

1. **多传感器融合**: 结合GPS、IMU、摄像头等多源数据
2. **机器学习**: 使用ML模型提高分类准确性
3. **实时预测**: 实现更长期的轨迹预测
4. **地图集成**: 结合高精度地图提高定位精度

## 故障排除

### 常见问题

1. **数据格式错误**: 确保CSV文件包含所有必需列
2. **坐标系统不一致**: 检查历史数据和当前数据的坐标系
3. **分类结果异常**: 调整阈值参数或检查数据质量
4. **匹配失败**: 检查轨迹数据质量和相似性

### 调试建议

1. **可视化检查**: 使用可视化工具检查轨迹形状
2. **参数调优**: 通过交叉验证调整分类参数
3. **数据清洗**: 移除异常数据点和噪声
4. **分步验证**: 逐步验证每个分析模块

## 联系信息

作者：MiniMax Agent  
版本：1.0  
更新日期：2025-12-15