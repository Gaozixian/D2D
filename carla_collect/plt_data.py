import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 将csv的数据画出来

def plot_csv_column(file_path, target_column, fig_size=(12, 6), dpi=100):
    """
    读取CSV目标列，按先后顺序绘制折线图
    :param file_path: CSV文件路径（相对/绝对路径）
    :param target_column: 目标列名（字符串）或列索引（整数，0开始）
    :param fig_size: 图片尺寸（宽, 高），默认(12,6)
    :param dpi: 图片清晰度，默认100
    """
    # ---------------------- 1. 读取CSV数据 ----------------------
    try:
        # 根据列名/列索引读取目标列（只加载目标列，提升效率）
        if isinstance(target_column, str):
            # 按列名读取（推荐，无需关心列位置）
            df = pd.read_csv(
                file_path,
                usecols=[target_column],  # 只加载目标列，减少内存占用
                encoding='utf-8',  # 中文适配，乱码时改为'gbk'
                # errors='ignore'  # 忽略特殊字符错误
            )
        elif isinstance(target_column, int):
            # 按列索引读取（无表头时使用）
            df = pd.read_csv(
                file_path,
                usecols=[target_column],
                header=None,  # 无表头时指定header=None
                encoding='utf-8',
            )
            # 给无表头的列命名（方便后续显示）
            df.columns = [f'列_{target_column}']
            target_column = f'列_{target_column}'  # 更新列名用于后续显示
        else:
            print("错误：target_column必须是列名（字符串）或列索引（整数）")
            return

        # 数据预处理：去除空值、转换为数值类型（避免绘图失败）
        df = df.dropna()  # 删除空值行
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce').dropna()  # 转换为数值型，失败值设为NaN后删除
        if df.empty:
            print(f"错误：目标列「{target_column}」无有效数值数据")
            return

        # ---------------------- 2. 绘制折线图 ----------------------
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示乱码（Windows）
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

        # 创建画布
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

        # 准备X轴（行索引=数据先后顺序，从1开始更直观）、Y轴（目标列数据）
        x_data = np.arange(1, len(df) + 1)  # X轴：1,2,3,...n（n为有效数据行数）
        y_data = df[target_column].values  # Y轴：目标列的数值数据

        # 绘制折线图（自定义样式：蓝色实线+圆点标记，线条宽度2）
        ax.plot(
            x_data, y_data,
            color='#2E86AB',  # 线条颜色（可替换为'red'/'green'等）
            linewidth=1.5,    # 线条宽度
            marker='.',       # 数据点标记（'.'=小点，'o'=圆点，None=无标记）
            markersize=3,     # 标记大小
            alpha=0.8         # 透明度（避免点重叠时看不清）
        )

        # ---------------------- 3. 图表美化（提升可读性） ----------------------
        ax.set_title(f'CSV列「{target_column}」数据趋势图（按先后顺序）', fontsize=16, pad=20)
        ax.set_xlabel('数据顺序（第n行）', fontsize=12)
        ax.set_ylabel(target_column, fontsize=12)  # Y轴标签用列名

        # 设置网格（便于读取数值）
        ax.grid(True, alpha=0.3, linestyle='--')

        # 调整坐标轴刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=10)

        # 自动调整布局（避免标签被截断）
        plt.tight_layout()

        # ---------------------- 4. 显示/保存图片 ----------------------
        plt.show()  # 显示图片（运行后弹出窗口）
        # 可选：保存图片（路径可自定义，支持png/jpg/pdf格式）
        # fig.savefig(f'{target_column}_trend.png', dpi=150, bbox_inches='tight')
        print(f"绘图完成！共处理{len(df)}行有效数据")

    except FileNotFoundError:
        print(f"错误：未找到文件「{file_path}」，请检查路径是否正确")
    except KeyError:
        print(f"错误：CSV文件中不存在列名「{target_column}」")
    except Exception as e:
        print(f"错误：绘图失败 → {str(e)}")

# ------------------- 调用示例（直接修改这部分！） -------------------
if __name__ == "__main__":
    # 1. 替换为你的CSV文件路径（相对路径/绝对路径均可）
    # 示例：相对路径（CSV和脚本在同一文件夹）→ "carla_data.csv"
    # 示例：子文件夹路径 → "carla_data_collect/your_data.csv"
    # 示例：Windows绝对路径 → "C:/Users/xxx/Desktop/carla_data.csv"
    csv_file_path = "/home/ubuntu123/carla_release/PythonAPI/examples/gaozixian/carla_data_collect/20251201_214342/csv/vehicle_data.csv"

    # 2. 替换为目标列（支持列名或列索引）
    speed_target_col = "speed_kmh"  # 推荐：按列名（如速度列、转向角列"steer"）
    # target_col = 3  # 备选：按列索引（3=第4列，无表头时使用）
    steer_target_col = "steer"
    

    # 执行绘图
    plot_csv_column(
        file_path=csv_file_path,
        target_column=speed_target_col,
        fig_size=(14, 7),  # 可选：调整图片大小
        dpi=120  # 可选：提高清晰度
    )
    plot_csv_column(
        file_path=csv_file_path,
        target_column=steer_target_col,
        fig_size=(14, 7),  # 可选：调整图片大小
        dpi=120  # 可选：提高清晰度
    )