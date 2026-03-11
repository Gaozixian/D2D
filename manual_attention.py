import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFilter


class AttentionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("环卫感知：人工注意力模拟器 (热力过渡优化版)")

        # --- 核心参数调节 ---
        self.radius = 45  # 注意力影响半径
        self.step_intensity = 15  # 单次点击增加的强度 (之前是35，现在调小使颜色变化变慢)
        self.max_alpha = 150  # 最高不透明度 (0-255)，确保始终能看清背景

        self.image_path = None
        self.tk_image = None
        self.original_img = None
        self.intensity_map = None
        self.attention_layer = None

        self.setup_ui()

    def setup_ui(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(side="top", fill="x", padx=10, pady=5)

        tk.Button(control_frame, text="1. 加载图片", command=self.load_image).pack(side="left", padx=5)
        tk.Button(control_frame, text="2. 清除标注", command=self.clear_attention).pack(side="left", padx=5)
        tk.Button(control_frame, text="3. 保存结果", command=self.save_result).pack(side="left", padx=5)

        self.label_info = tk.Label(self.root, text="逻辑：点击次数越多，颜色从 绿 -> 黄 -> 红 缓慢过渡。")
        self.label_info.pack(pady=5)

        self.canvas = tk.Canvas(self.root, cursor="target", bg="gray30")
        self.canvas.pack(expand=True, fill="both")
        self.canvas.bind("<Button-1>", self.add_attention)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if self.image_path:
            img = Image.open(self.image_path).convert("RGBA")
            img.thumbnail((1000, 800))
            self.original_img = img
            self.clear_attention()

    def clear_attention(self):
        if self.original_img:
            # 初始化强度图
            self.intensity_map = Image.new("L", self.original_img.size, 0)
            self.attention_layer = Image.new("RGBA", self.original_img.size, (0, 0, 0, 0))
            self.update_canvas()

    def add_attention(self, event):
        if self.original_img:
            # 1. 生成渐变笔触
            brush_size = self.radius * 2
            brush = Image.new("L", (brush_size * 2, brush_size * 2), 0)
            draw = ImageDraw.Draw(brush)

            for i in range(brush_size, 0, -2):
                # 强度衰减公式，使边缘更柔和
                power = int(self.step_intensity * (1 - i / brush_size))
                draw.ellipse([brush_size - i, brush_size - i, brush_size + i, brush_size + i], fill=power)

            brush = brush.filter(ImageFilter.GaussianBlur(radius=8))

            # 2. 叠加到主强度图 (使用 numpy 避免溢出并加速)
            temp_layer = Image.new("L", self.intensity_map.size, 0)
            temp_layer.paste(brush, (event.x - brush_size, event.y - brush_size))

            arr_total = np.array(self.intensity_map).astype(np.int32)  # 使用 int32 累加
            arr_new = np.array(temp_layer).astype(np.int32)

            combined = np.clip(arr_total + arr_new, 0, 255).astype(np.uint8)
            self.intensity_map = Image.fromarray(combined)

            # 3. 映射伪彩色
            self.apply_heatmap_logic(combined)
            self.update_canvas()

    def apply_heatmap_logic(self, intensity_array):
        """将强度映射为 绿-黄-红，解决 uint8 510 溢出问题"""
        # 关键修复：转为 float32 处理计算
        data = intensity_array.astype(np.float32)
        h, w = data.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # 颜色映射曲线：
        # R: 0 -> 255 (随强度上升)
        rgba[..., 0] = np.clip(data * 2, 0, 255).astype(np.uint8)
        # G: 255 -> 0 (在强度超过一半后开始下降)
        rgba[..., 1] = np.clip(510 - data * 2, 0, 255).astype(np.uint8)
        # B: 始终为 0
        rgba[..., 2] = 0
        # A: 透明度平滑上升，最高锁定在 self.max_alpha
        rgba[..., 3] = np.where(data > 0, np.clip(data * 1.2 + 40, 0, self.max_alpha), 0).astype(np.uint8)

        self.attention_layer = Image.fromarray(rgba, "RGBA")

    def update_canvas(self):
        combined = Image.alpha_composite(self.original_img, self.attention_layer)
        self.tk_image = ImageTk.PhotoImage(combined)
        self.canvas.config(width=combined.width, height=combined.height)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def save_result(self):
        if self.original_img and self.attention_layer:
            save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG", "*.png")])
            if save_path:
                final_out = Image.alpha_composite(self.original_img, self.attention_layer).convert("RGB")
                final_out.save(save_path)
                print(f"结果已保存: {save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = AttentionApp(root)
    root.mainloop()