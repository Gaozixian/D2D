# 负责绘制加速度和速度的变化曲线
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# 设置中文字体和图表样式
# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False
# 1. 读取数据
df = pd.read_csv('global_vehicle_data.csv')

# 转换相对时间（以秒为单位，从0开始）
df['time_rel'] = df['timestamp'] - df['timestamp'].iloc[0]

# 2. 计算合成标量 (Magnitudes)
# 加速度合成：sqrt(ax^2 + ay^2 + az^2)
df['a_mag'] = np.sqrt(df['acceleration_x']**2 + df['acceleration_y']**2 + df['acceleration_z']**2)
# 速度合成：sqrt(vx^2 + vy^2 + vz^2)
df['v_mag'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2 + df['velocity_z']**2)

# 3. 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# --- 子图 1: 各个分量 ---
# 绘制加速度分量 (虚线)
ax1.plot(df['time_rel'], df['acceleration_x'], label='Acc X', linestyle='--', alpha=0.7)
ax1.plot(df['time_rel'], df['acceleration_y'], label='Acc Y', linestyle='--', alpha=0.7)
ax1.plot(df['time_rel'], df['acceleration_z'], label='Acc Z', linestyle='--', alpha=0.7)
# 绘制速度分量 (实线)
ax1.plot(df['time_rel'], df['velocity_x'], label='Vel X', linewidth=1.5)
ax1.plot(df['time_rel'], df['velocity_y'], label='Vel Y', linewidth=1.5)
ax1.plot(df['time_rel'], df['velocity_z'], label='Vel Z', linewidth=1.5)

ax1.set_title('速度加速度分解图')
ax1.set_ylabel('速度/加速度')
ax1.legend(loc='upper right', ncol=2)
ax1.grid(True, linestyle=':', alpha=0.6)

# --- 子图 2: 合成标量 ---
ax2.plot(df['time_rel'], df['a_mag'], label='加速度', color='red', linewidth=2)
ax2.plot(df['time_rel'], df['v_mag'], label='速度', color='blue', linewidth=2)

ax2.set_title('加速度和速度的标量大小')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('速度/加速度')
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('vehicle_dynamics_plot.png')
print("曲线图已保存为 vehicle_dynamics_plot.png")