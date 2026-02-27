import torch
import torch.nn as nn
import math
from Backbone import create_resnet_cbam  # 导入底层骨干网络


class MotionLSTMEncoder(nn.Module):
    """LSTM 处理历史状态"""

    def __init__(self, input_size=10, hidden_size=512, num_layers=2):
        super(MotionLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.ln(last_hidden)


class PositionalEncoding(nn.Module):
    """标准的正弦/余弦时序位置编码"""

    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class DualStreamDrivingModel(nn.Module):
    """双流端到端自动驾驶模型核心"""

    def __init__(self, use_shared_weights=True, visual_d_model=512, lstm_hidden_size=512):
        super(DualStreamDrivingModel, self).__init__()

        self.use_shared_weights = use_shared_weights
        self.visual_d_model = visual_d_model

        # ==================== 1. 视觉流 (Cross-Attention 架构) ====================
        self.resnet = create_resnet_cbam()
        self.feature_proj = nn.Linear(2048, visual_d_model)
        self.pos_encoder = PositionalEncoding(visual_d_model, max_len=500)

        # 显式定义交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=visual_d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 为了网络稳定性，加入标准的 Transformer 前馈网络 (FFN) 和 LayerNorm
        self.vis_ln1 = nn.LayerNorm(visual_d_model)
        self.vis_ln2 = nn.LayerNorm(visual_d_model)
        self.vis_ffn = nn.Sequential(
            nn.Linear(visual_d_model, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, visual_d_model)
        )

        # ==================== 2. 状态流 ====================
        self.lstm_hidden_size = lstm_hidden_size
        self.motion_lstm = MotionLSTMEncoder(
            input_size=10,
            hidden_size=lstm_hidden_size,
            num_layers=2
        )

        # ==================== 3. 融合模块 (MLP) ====================
        fusion_input_dim = visual_d_model + lstm_hidden_size
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, lstm_hidden_size),
            nn.ReLU()
        )

        # ==================== 4. 预测控制头 ====================
        self.control_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # [accel_x, accel_y, accel_z, steer]
        )

    def extract_image_features(self, img_t_minus_2, img_t_minus_1, img_t):
        f1 = self.resnet.get_feature_maps(img_t_minus_2)
        f2 = self.resnet.get_feature_maps(img_t_minus_1)
        f3 = self.resnet.get_feature_maps(img_t)
        return f1, f2, f3

    def forward(self, img_t_minus_2, img_t_minus_1, img_t, state_history):
        batch_size = img_t.size(0)

        # ---------------- A. 视觉流处理 (Cross-Attention) ----------------
        f1, f2, f3 = self.extract_image_features(img_t_minus_2, img_t_minus_1, img_t)

        def flatten_and_project(f):
            # (Batch, 2048, H, W) -> (Batch, H*W, 512)
            flat = f.view(batch_size, f.size(1), -1).transpose(1, 2)
            return self.feature_proj(flat)

        proj_1 = flatten_and_project(f1)  # t-2 特征
        proj_2 = flatten_and_project(f2)  # t-1 特征
        proj_3 = flatten_and_project(f3)  # t   特征

        # 1. 构造 K 和 V：拼接 t-2 和 t-1 时刻的历史特征
        # 假设分辨率处理后是 7x7=49 个 Patch，这里拼接后是 98 个 Patch
        kv_features = torch.cat([proj_1, proj_2], dim=1)
        kv_features = self.pos_encoder(kv_features)  # 添加位置编码

        # 2. 构造 Q：当前时刻 t 的特征
        q_features = proj_3
        q_features = self.pos_encoder(q_features)  # 添加位置编码

        # 3. 交叉注意力机制 (Q 查 K/V)
        # q_features 形状: (Batch, 49, d_model)
        # kv_features 形状: (Batch, 98, d_model)
        attended_output, _ = self.cross_attention(
            query=q_features,
            key=kv_features,
            value=kv_features
        )

        # 4. 残差连接与前馈网络 (标准的 Transformer 解码块逻辑)
        # 让当前帧的原始特征与“查找到的历史特征”相加
        vis_out = self.vis_ln1(q_features + attended_output)
        vis_out = self.vis_ln2(vis_out + self.vis_ffn(vis_out))

        # 5. 全局池化得到视觉特征向量
        # vis_out 的形状是 (Batch, 49, d_model)，平均后变为 (Batch, d_model)
        visual_vector = vis_out.mean(dim=1)

        # ---------------- B. 状态流处理 ----------------
        lstm_vector = self.motion_lstm(state_history)

        # ---------------- C. 融合逻辑 ----------------
        # Output = MLP(Concat(Visual, LSTM)) + LSTM
        combined_features = torch.cat([visual_vector, lstm_vector], dim=1)
        mlp_output = self.fusion_mlp(combined_features)
        fused_final = mlp_output + lstm_vector

        # ---------------- D. 预测并输出 ----------------
        prediction = self.control_head(fused_final)
        return prediction