````##  carla_collect代码文件说明
carla_collect/carla_data_collect：负责四个视角和控制信息的收集

carla_collect/my_manual_control.py：负责读取生成的车辆，接着在车辆周围放置四个摄像头

carla_collect/manual_master.py： 方向盘控制车辆

carla_collect/manual_control.py：carla自带的控制车辆，这里修改了使其可以生成特斯拉车辆

carla_collect/generate_vehicles.py：生成随机车辆

main：模型训练文件

## carla_location代码文件说明
ipynb： 负责绘制两个路径的对比图片以及转向判定对比图

## carla_visual代码文件说明

## carla_LSTM代码文件说明

## carla仿真启动说明
版本：	
	carla 0.9.15
	UE 4.26
	python 3.7.8
启动顺序如下
```commandline
打开UE4界面
cd ~/carla/Unreal/CarlaUE4
~/UnrealEngine_4.26/Engine/Binaries/Linux/UE4Editor "$PWD/CarlaUE4.uproject" -game -quality-level=Low -uncooked -opengl3 -windowed -ResX=2000 -ResY=2000

另开终端
cd ~/carla/PythonAPI/util
python3 config.py --map Town07	# 切换地图
注意：如果该地图是第一次加载，则会进行编译，编译过程中可能会发生崩溃，多试几次就行
3. 生成自车
cd ~/gzx/D2D/carla_collect
python3 manual_control.py	# 生成自车
python3 manual_control.py --filter vehicle.tesla.cybertruck

4. 开始数据采集
cd ~/gaozixian/D2D/carla_collect
python3 my_manual_control.py # 开始数据采集
python3 automic.py # 可视化界面
```




```python
```
````