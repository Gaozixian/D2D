import carla
import cv2
import numpy as np
import time
import csv
import os
import pygame
import queue

# 初始化pygame
pygame.init()
display = pygame.display.set_mode((800, 600))

# 连接到Carla服务器
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# 获取世界和地图
world = client.load_world('Town07')
map = world.get_map()

# 设置天气
weather = carla.WeatherParameters.ClearNoon
world.set_weather(weather)

# 获取蓝图库
blueprint_library = world.get_blueprint_library()

# 选择车辆蓝图
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

# 选择一个初始生成点
spawn_points = map.get_spawn_points()
spawn_point = spawn_points[0]

# 生成车辆
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# 设置前视摄像头蓝图
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '110')

# 设置摄像头的位置
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

# 生成摄像头
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# 初始化数据存储列表
data = []

# 定义图像保存路径
image_folder = 'images'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# 定义CSV文件路径
csv_file = 'driving_data.csv'

# 创建一个队列用于传递图像数据
image_queue = queue.Queue()

# 定义摄像头数据处理函数
def process_image(image):
    img = np.array(image.raw_data)
    img = img.reshape((image.height, image.width, 4))
    img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色空间以适配pygame
    timestamp = time.time()
    image_name = f'{image_folder}/image_{timestamp}.png'
    cv2.imwrite(image_name, img)

    # 获取速度
    velocity = vehicle.get_velocity()
    speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    # 获取加速度
    acceleration = vehicle.get_acceleration()

    # 获取方向盘转角
    control = vehicle.get_control()
    steering_angle = control.steer

    # 保存数据
    data.append([image_name, speed, acceleration.x, acceleration.y, acceleration.z, steering_angle])

    # 保存数据到CSV文件
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['image_name', 'speed', 'acceleration_x', 'acceleration_y', 'acceleration_z', 'steering_angle'])
        writer.writerow([image_name, speed, acceleration.x, acceleration.y, acceleration.z, steering_angle])

    # 将图像数据放入队列
    image_queue.put(img)

# 附加回调函数到摄像头
camera.listen(process_image)

# 车辆控制参数
throttle = 0.0
steer = 0.0
brake = 0.0

try:
    while True:
        # 处理pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    throttle = 0.5
                elif event.key == pygame.K_DOWN:
                    brake = 1.0
                elif event.key == pygame.K_LEFT:
                    steer = -0.5
                elif event.key == pygame.K_RIGHT:
                    steer = 0.5
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    throttle = 0.0
                elif event.key == pygame.K_DOWN:
                    brake = 0.0
                elif event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    steer = 0.0

        # 应用控制到车辆
        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steer
        control.brake = brake
        vehicle.apply_control(control)

        # 从队列中获取图像数据并更新显示
        if not image_queue.empty():
            img = image_queue.get()
            pygame.surfarray.blit_array(display, img.swapaxes(0, 1))
            pygame.display.flip()

        time.sleep(0.01)

except KeyboardInterrupt:
    print('数据采集停止')
finally:
    # 销毁车辆和摄像头
    if vehicle is not None:
        vehicle.destroy()
    if camera is not None:
        camera.destroy()
    # 退出pygame
    pygame.quit()

