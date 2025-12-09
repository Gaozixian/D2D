## 代码文件说明
carla_collect/carla_data_collect：负责四个视角和控制信息的收集

carla_collect/my_manual_control.py：负责读取生成的车辆，接着在车辆周围放置四个摄像头

carla_collect/manual_master.py： 方向盘控制车辆

carla_collect/manual_control.py：carla自带的控制车辆

carla_collect/generate_vehicles.py：生成随机车辆

## carla仿真说明
```commandline
cd ~/carla/Unreal/CarlaUE4  # 进入carla目录
~/UnrealEngine_4.26/Engine/Binaries/Linux/UE4Editor "$PWD/CarlaUE4.uproject"    # 打开UE4carla客户端
```

```python
```
1. 启动carla