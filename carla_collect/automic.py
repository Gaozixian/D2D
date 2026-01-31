#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Carla Autonomous Vehicle Control Application

功能说明:
1. 连接到Carla仿真器
2. 生成指定类型的车辆
3. 通过按钮控制车辆自动驾驶（基于PID控制器）
4. 实时显示车辆状态
5. 安全的资源清理机制

作者: MiniMax Agent
"""

import carla
import random
import time
import threading
import math
import tkinter as tk
from tkinter import messagebox, ttk


# ============================================================================
# PID控制器类 - 用于自动驾驶的横向和纵向控制
# ============================================================================

class PIDController:
    """
    PID控制器类，负责计算车辆的控制量（转向、油门/刹车）

    横向控制（Steering）: 基于当前车辆朝向与目标路径点的角度偏差
    纵向控制（Throttle/Brake）: 基于当前车速与目标车速的速度偏差
    """

    def __init__(self, vehicle, lateral_params, longitudinal_params, target_speed=30.0):
        """
        初始化PID控制器

        Args:
            vehicle: Carla车辆对象
            lateral_params: 横向PID参数，字典格式 {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0, 'dt': 0.03}
            longitudinal_params: 纵向PID参数，字典格式
            target_speed: 目标车速（km/h）
        """
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.map = self.world.get_map()
        self.target_speed = target_speed

        # PID参数
        self.lat_params = lateral_params
        self.lon_params = longitudinal_params

        # 误差历史缓冲区（用于积分和微分计算）
        self._lat_error_buffer = []
        self._lon_error_buffer = []

        # 缓冲区最大长度
        self._buffer_max_len = 50

    def run_step(self):
        """
        执行一步控制计算

        Returns:
            carla.VehicleControl: 车辆控制命令
            float: 当前车速（km/h）
        """
        # 获取车辆当前状态
        transform = self.vehicle.get_transform()
        location = transform.location
        velocity = self.vehicle.get_velocity()

        # 计算当前车速（m/s转km/h）
        current_speed = 3.6 * math.sqrt(
            velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
        )

        # 获取目标路径点（前瞻距离根据车速动态调整）
        current_waypoint = self.map.get_waypoint(location)
        lookahead_distance = max(5.0, current_speed * 0.5 + 3.0)

        try:
            next_waypoints = current_waypoint.next(lookahead_distance)
            target_waypoint = next_waypoints[0] if next_waypoints else current_waypoint
        except:
            target_waypoint = current_waypoint

        # 计算横向控制（转向）
        steer = self._calculate_steering(target_waypoint, transform)

        # 计算纵向控制（油门/刹车）
        throttle_or_brake = self._calculate_throttle_brake(current_speed)

        # 构建控制命令
        control = carla.VehicleControl()
        control.steer = steer

        if throttle_or_brake >= 0:
            control.throttle = min(throttle_or_brake, 1.0)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(throttle_or_brake), 1.0)

        return control, current_speed

    def _calculate_steering(self, target_waypoint, vehicle_transform):
        """
        计算转向角度

        Args:
            target_waypoint: 目标路径点
            vehicle_transform: 车辆变换信息

        Returns:
            float: 转向值（-1.0到1.0）
        """
        # 获取车辆朝向向量
        vehicle_forward = vehicle_transform.get_forward_vector()

        # 计算指向目标的向量
        target_location = target_waypoint.transform.location
        to_target = target_location - vehicle_transform.location
        to_target.z = 0  # 忽略高度差
        to_target_length = to_target.length()

        if to_target_length < 0.001:
            return 0.0

        to_target = to_target / to_target_length  # 归一化

        # 计算角度偏差
        # 叉积判断左右
        cross_product = vehicle_forward.x * to_target.y - vehicle_forward.y * to_target.x

        # 点积计算夹角
        dot_product = (vehicle_forward.x * to_target.x +
                       vehicle_forward.y * to_target.y)
        dot_product = max(-1.0, min(1.0, dot_product))

        angle = math.acos(dot_product)

        # 根据叉积符号确定转向方向
        if cross_product < 0:
            angle = -angle

        # 更新误差缓冲区
        self._lat_error_buffer.append(angle)
        if len(self._lat_error_buffer) > self._buffer_max_len:
            self._lat_error_buffer.pop(0)

        # PID计算
        return self._pid_compute(
            self._lat_error_buffer,
            self.lat_params['K_P'],
            self.lat_params['K_I'],
            self.lat_params['K_D'],
            self.lat_params['dt']
        )

    def _calculate_throttle_brake(self, current_speed):
        """
        计算油门或刹车控制量

        Args:
            current_speed: 当前车速（km/h）

        Returns:
            float: 正值为油门（0-1），负值为刹车（0-1）
        """
        # 速度误差
        error = self.target_speed - current_speed

        # 更新误差缓冲区
        self._lon_error_buffer.append(error)
        if len(self._lon_error_buffer) > self._buffer_max_len:
            self._lon_error_buffer.pop(0)

        # PID计算
        return self._pid_compute(
            self._lon_error_buffer,
            self.lon_params['K_P'],
            self.lon_params['K_I'],
            self.lon_params['K_D'],
            self.lon_params['dt']
        )

    def _pid_compute(self, error_buffer, kp, ki, kd, dt):
        """
        PID计算核心函数

        Args:
            error_buffer: 误差历史缓冲区
            kp: 比例增益
            ki: 积分增益
            kd: 微分增益
            dt: 时间步长

        Returns:
            float: PID输出值
        """
        if len(error_buffer) < 2:
            return kp * error_buffer[-1] if error_buffer else 0.0

        # 比例项
        p_term = kp * error_buffer[-1]

        # 积分项
        i_term = ki * sum(error_buffer) * dt

        # 微分项
        d_term = kd * (error_buffer[-1] - error_buffer[-2]) / dt

        return p_term + i_term + d_term

    def set_target_speed(self, speed):
        """设置目标车速（km/h）"""
        self.target_speed = speed


# ============================================================================
# Carla自动驾驶应用主类
# ============================================================================

class CarlaAutoPilotApp:
    """
    Carla自动驾驶控制应用程序主类

    提供图形界面用于：
    - 连接Carla服务器
    - 生成车辆
    - 启动/停止自动驾驶
    - 显示车辆状态
    """

    def __init__(self, root):
        """
        初始化应用程序

        Args:
            root: Tkinter根窗口
        """
        self.root = root
        self.root.title("Carla Autonomous Vehicle Controller")
        self.root.geometry("450x400")
        self.root.resizable(False, False)

        # Carla相关变量
        self.client = None
        self.world = None
        self.map = None
        self.vehicle = None
        self.controller = None

        # 控制状态
        self.autopilot_enabled = False
        self.running = True

        # 创建界面
        self._create_widgets()

        # 启动控制循环线程
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

    def _create_widgets(self):
        """创建GUI组件"""

        # 样式配置
        style = ttk.Style()
        style.configure('TLabel', font=('Microsoft YaHei UI', 10))
        style.configure('TButton', font=('Microsoft YaHei UI', 10))

        # 1. 连接设置区域
        frame_connection = ttk.LabelFrame(self.root, text="Connection", padding=10)
        frame_connection.pack(fill="x", padx=10, pady=5)

        self.btn_connect = ttk.Button(
            frame_connection,
            text="Connect to Carla Server",
            command=self._connect_to_carla
        )
        self.btn_connect.pack(fill="x", pady=5)

        self.lbl_connection_status = ttk.Label(
            frame_connection,
            text="Status: Disconnected",
            foreground="red"
        )
        self.lbl_connection_status.pack()

        # 2. 车辆生成区域
        frame_spawn = ttk.LabelFrame(self.root, text="Vehicle Spawn", padding=10)
        frame_spawn.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_spawn, text="Blueprint Filter:").pack(side="left", padx=5)

        self.entry_blueprint = ttk.Entry(frame_spawn, width=25)
        self.entry_blueprint.insert(0, "vehicle.tesla.model3")
        self.entry_blueprint.pack(side="left", padx=5)

        self.btn_spawn = ttk.Button(
            frame_spawn,
            text="Spawn Vehicle",
            command=self._spawn_vehicle,
            state="disabled"
        )
        self.btn_spawn.pack(side="left", padx=5)

        # 3. 自动驾驶控制区域
        frame_control = ttk.LabelFrame(self.root, text="Autopilot Control", padding=10)
        frame_control.pack(fill="x", padx=10, pady=5)

        # 目标速度设置
        ttk.Label(frame_control, text="Target Speed (km/h):").pack(side="left", padx=5)

        self.spin_speed = ttk.Spinbox(
            frame_control,
            from_=10,
            to=200,
            increment=5,
            width=8
        )
        self.spin_speed.set(50)
        self.spin_speed.pack(side="left", padx=5)

        # 自动驾驶开关按钮
        self.btn_autopilot = ttk.Button(
            frame_control,
            text="START AUTOPILOT",
            command=self._toggle_autopilot,
            state="disabled"
        )
        self.btn_autopilot.pack(fill="x", pady=10)

        # 4. 状态显示区域
        frame_status = ttk.LabelFrame(self.root, text="Status", padding=10)
        frame_status.pack(fill="x", padx=10, pady=5)

        self.lbl_speed = ttk.Label(
            frame_status,
            text="Speed: 0.0 km/h",
            font=('Microsoft YaHei UI', 14, 'bold')
        )
        self.lbl_speed.pack(pady=5)

        self.lbl_mode = ttk.Label(
            frame_status,
            text="Mode: Manual",
            font=('Microsoft YaHei UI', 12)
        )
        self.lbl_mode.pack(pady=5)

        # 5. 退出按钮
        ttk.Button(
            self.root,
            text="Exit & Cleanup",
            command=self._on_close
        ).pack(pady=10)

        # 窗口关闭事件处理
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _connect_to_carla(self):
        """连接到Carla服务器"""
        try:
            # 创建Carla客户端
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(5.0)

            # 获取世界和地图
            self.world = self.client.get_world()
            self.map = self.world.get_map()

            # 更新界面状态
            self.lbl_connection_status.config(
                text="Status: Connected",
                foreground="green"
            )
            self.btn_connect.config(state="disabled")
            self.btn_spawn.config(state="normal")

            messagebox.showinfo("Success", "Successfully connected to Carla server!")

        except Exception as e:
            messagebox.showerror(
                "Connection Error",
                f"Failed to connect to Carla server:\n{str(e)}\n\n"
                "Please make sure the Carla server is running on localhost:2000"
            )

    def _spawn_vehicle(self):
        """生成车辆"""
        # 如果已存在车辆，先销毁
        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
                self.vehicle = None
            except:
                pass

        try:
            # 获取蓝图库
            blueprint_library = self.world.get_blueprint_library()

            # 获取用户指定的车辆类型
            filter_text = self.entry_blueprint.get().strip()
            blueprints = blueprint_library.filter(filter_text)

            if not blueprints:
                messagebox.showwarning(
                    "Warning",
                    f"No vehicle blueprints found matching: '{filter_text}'\n\n"
                    "Try using 'vehicle.*' or specific vehicle names like:\n"
                    "- vehicle.tesla.model3\n"
                    "- vehicle.nissan.patrol\n"
                    "- vehicle.audi.etron"
                )
                return

            # 选择第一个匹配的蓝图
            blueprint = blueprints[0]

            # 获取随机生成点
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()

            # 生成车辆
            self.vehicle = self.world.spawn_actor(blueprint, spawn_point)

            # 配置PID控制器
            # 横向控制参数（转向）
            lateral_params = {
                'K_P': 1.5,  # 比例增益
                'K_D': 0.3,  # 微分增益
                'K_I': 0.05,  # 积分增益
                'dt': 0.05  # 时间步长
            }

            # 纵向控制参数（油门/刹车）
            longitudinal_params = {
                'K_P': 0.5,
                'K_D': 0.1,
                'K_I': 0.02,
                'dt': 0.05
            }

            # 获取目标速度
            target_speed = float(self.spin_speed.get())

            # 创建控制器
            self.controller = PIDController(
                self.vehicle,
                lateral_params,
                longitudinal_params,
                target_speed
            )

            # 移动观察者视角到车辆位置
            self._move_spectator_to_vehicle()

            # 更新界面状态
            self.btn_autopilot.config(state="normal")
            messagebox.showinfo(
                "Success",
                f"Vehicle spawned successfully!\n"
                f"Blueprint: {blueprint.id}\n"
                f"Target Speed: {target_speed} km/h"
            )

        except Exception as e:
            messagebox.showerror("Spawn Error", f"Failed to spawn vehicle:\n{str(e)}")

    def _move_spectator_to_vehicle(self):
        """移动观察者视角到车辆位置"""
        try:
            spectator = self.world.get_spectator()
            transform = self.vehicle.get_transform()

            # 设置观察者在车辆后上方
            spectator_transform = carla.Transform(
                transform.location + carla.Location(z=15),
                carla.Rotation(pitch=-60, yaw=transform.rotation.yaw)
            )

            spectator.set_transform(spectator_transform)

        except Exception as e:
            print(f"Warning: Failed to move spectator: {e}")

    def _toggle_autopilot(self):
        """切换自动驾驶模式"""
        if not self.vehicle:
            messagebox.showwarning("Warning", "No vehicle available!")
            return

        self.autopilot_enabled = not self.autopilot_enabled

        if self.autopilot_enabled:
            # 更新目标速度
            target_speed = float(self.spin_speed.get())
            self.controller.set_target_speed(target_speed)

            # 更新按钮状态
            self.btn_autopilot.config(text="STOP AUTOPILOT")
            self.lbl_mode.config(text="Mode: Autopilot", foreground="green")

        else:
            # 停止自动驾驶，施加刹车
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, brake=1.0)
            )

            # 更新按钮状态
            self.btn_autopilot.config(text="START AUTOPILOT")
            self.lbl_mode.config(text="Mode: Manual", foreground="black")

    def _control_loop(self):
        """
        自动驾驶控制循环（后台线程运行）

        以20Hz的频率运行PID控制器
        """
        while self.running:
            try:
                if (self.vehicle and
                        self.vehicle.is_alive and
                        self.autopilot_enabled and
                        self.controller):

                    # 执行PID控制
                    control, current_speed = self.controller.run_step()

                    # 应用控制
                    self.vehicle.apply_control(control)

                    # 更新UI显示
                    self.root.after(
                        0,
                        lambda s=current_speed: self.lbl_speed.config(
                            text=f"Speed: {s:.1f} km/h"
                        )
                    )

                elif self.vehicle and self.vehicle.is_alive:
                    # 仅显示速度（手动模式）
                    velocity = self.vehicle.get_velocity()
                    speed = 3.6 * math.sqrt(
                        velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
                    )

                    self.root.after(
                        0,
                        lambda s=speed: self.lbl_speed.config(
                            text=f"Speed: {s:.1f} km/h"
                        )
                    )

            except Exception as e:
                print(f"Control loop error: {e}")

            # 控制频率：20Hz
            time.sleep(0.05)

    def _on_close(self):
        """窗口关闭处理"""
        self.running = False

        # 销毁车辆
        if self.vehicle:
            try:
                print("Destroying vehicle...")
                self.vehicle.destroy()
            except:
                pass

        # 关闭窗口
        self.root.destroy()


# ============================================================================
# 主程序入口
# ============================================================================

def main():
    """主函数"""
    # 检查Carla依赖
    try:
        import carla
    except ImportError:
        print("Error: Carla library not found!")
        print("Please install the Carla client library:")
        print("  pip install carla")
        return

    # 创建Tkinter根窗口
    root = tk.Tk()

    # 设置字体（支持中文）
    try:
        root.tk.call('encoding', 'system')
    except:
        pass

    # 创建应用程序
    app = CarlaAutoPilotApp(root)

    # 运行主循环
    root.mainloop()


if __name__ == "__main__":
    main()