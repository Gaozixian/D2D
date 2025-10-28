import carla
import random
import time

# 连接到CARLA服务器
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# 切换到指定地图
world = client.load_world('Town07')


# 获取当前世界
world = client.get_world()

# 设置同步模式（可选，确保精确控制）
settings = world.get_settings()
settings.synchronous_mode = True  # 打开同步模式
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# 获取蓝图库
blueprint_library = world.get_blueprint_library()

# 获取车辆蓝图
vehicle_blueprints = blueprint_library.filter('vehicle.*')

# 获取所有生成点
spawn_points = world.get_map().get_spawn_points()

# 添加随机车辆
number_of_vehicles = 50  # 要生成的车辆数量
vehicles = []  # 存储生成的车辆
for _ in range(number_of_vehicles):
    # 随机选择一个车辆蓝图和生成点
    vehicle_bp = random.choice(vehicle_blueprints)
    spawn_point = random.choice(spawn_points)
    
    # 生成车辆
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle:  # 确保生成成功
        vehicles.append(vehicle)

# 设置每辆车的速度
traffic_manager = client.get_trafficmanager(8000)
traffic_manager.set_synchronous_mode(True)  # 配合同步模式
for vehicle in vehicles:
    # 随机设置目标速度（单位：km/h）
    target_speed = random.uniform(50, 60)
    vehicle.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.vehicle_percentage_speed_difference(vehicle, 100 - target_speed)
    traffic_manager.auto_lane_change(vehicle, True)  # 允许车辆自动变道

print(f"成功生成了 {len(vehicles)} 辆车")

# 模拟一段时间
try:
    for _ in range(3000):  # 运行200帧
        world.tick()  # 如果是同步模式，需要手动调用tick
        time.sleep(0.05)
finally:
    # 清理生成的车辆
    #for vehicle in vehicles:
    #    vehicle.destroy()
    print("所有车辆已销毁")
