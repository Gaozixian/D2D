import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from DPAI_Backbone import create_interaction_backbone


# ==================== 1. 基础组件 ====================
class MotionLSTMEncoder(nn.Module):
    def __init__(self, input_size=10, hidden_size=512, num_layers=2):
        super(MotionLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.ln(h_n[-1])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term);
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ==================== 2. DPAI 2.0 交互模块 ====================
class AdaptiveBidirectionalInteraction(nn.Module):
    def __init__(self, s_channels, d_channels, reduction=16):
        super(AdaptiveBidirectionalInteraction, self).__init__()
        self.s_conv7x7 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.s_to_d_mlp = nn.Sequential(nn.Linear(1, d_channels // reduction), nn.ReLU(inplace=True),
                                        nn.Linear(d_channels // reduction, d_channels))
        self.d_shared_mlp = nn.Sequential(nn.Linear(d_channels, d_channels // reduction, bias=False),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(d_channels // reduction, d_channels, bias=False))
        self.d_to_s_conv = nn.Sequential(nn.Conv2d(d_channels, 256, 1, bias=False), nn.BatchNorm2d(256),
                                         nn.ReLU(inplace=True), nn.Conv2d(256, 1, 1, bias=False), nn.Sigmoid())
        self.sigmoid = nn.Sigmoid()

    def forward(self, Fs, Fd, return_attn=False):
        avg_p = F.adaptive_avg_pool2d(Fd, 1).view(Fd.size(0), -1)
        max_p = F.adaptive_max_pool2d(Fd, 1).view(Fd.size(0), -1)
        d_stats = self.d_shared_mlp(avg_p) + self.d_shared_mlp(max_p)
        Ac = self.sigmoid(d_stats)
        Gds = self.d_to_s_conv(Fd * Ac.view(Fd.size(0), Fd.size(1), 1, 1))
        Gds_up = F.interpolate(Gds, size=(Fs.size(2), Fs.size(3)), mode='bilinear', align_corners=False)
        s_desc = torch.cat([torch.mean(Fs, dim=1, keepdim=True), torch.max(Fs, dim=1, keepdim=True)[0]], dim=1)

        # 这里的 As 就是我们要可视化的空间注意力图
        As = self.sigmoid(self.s_conv7x7(s_desc * Gds_up))

        Fs_out = Fs * As
        Gsd = self.s_to_d_mlp(F.adaptive_avg_pool2d(As, 1).view(As.size(0), -1))
        Fd_out = Fd * self.sigmoid(d_stats + Gsd).view(Fd.size(0), Fd.size(1), 1, 1)

        if return_attn:
            return Fs_out, Fd_out, As
        return Fs_out, Fd_out


class VisualInteractionTransformer(nn.Module):
    def __init__(self, d_model=512):
        super(VisualInteractionTransformer, self).__init__()
        self.backbone = create_interaction_backbone()
        self.interaction = AdaptiveBidirectionalInteraction(s_channels=512, d_channels=2048)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        self.proj = nn.Conv2d(2048, d_model, kernel_size=1)
        self.pos_encoder = PositionalEncoding(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.vis_ln1 = nn.LayerNorm(d_model)
        self.vis_ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model))
        self.vis_ln2 = nn.LayerNorm(d_model)

    def _get_enhanced_features(self, img, return_attn=False):
        fs, fd = self.backbone(img)
        if return_attn:
            _, fd_enhanced, attn = self.interaction(fs, fd, return_attn=True)
        else:
            _, fd_enhanced = self.interaction(fs, fd)

        combined_fd = fd + fd_enhanced
        combined_fd = self.fusion_conv(combined_fd)
        feat = self.proj(combined_fd)

        if return_attn:
            return feat.flatten(2).transpose(1, 2), attn
        return feat.flatten(2).transpose(1, 2)

    def forward(self, img_t2, img_t1, img_t, return_attn=False):
        if return_attn:
            f_t2 = self._get_enhanced_features(img_t2)
            f_t1 = self._get_enhanced_features(img_t1)
            f_t, attn = self._get_enhanced_features(img_t, return_attn=True)
        else:
            f_t2, f_t1, f_t = self._get_enhanced_features(img_t2), self._get_enhanced_features(
                img_t1), self._get_enhanced_features(img_t)

        q = self.pos_encoder(f_t)
        kv = self.pos_encoder(torch.cat([f_t2, f_t1], dim=1))
        attended_out, _ = self.cross_attention(q, kv, kv)
        out = self.vis_ln1(q + attended_out)
        out = self.vis_ln2(out + self.vis_ffn(out))

        if return_attn:
            return out.mean(dim=1), attn
        return out.mean(dim=1)


# ==================== 4. 主模型 ====================
class AutoHuberModel(nn.Module):
    def __init__(self, state_dim=10, vis_d_model=512):
        super(AutoHuberModel, self).__init__()

        self.visual_stream = VisualInteractionTransformer(vis_d_model)
        self.state_stream = MotionLSTMEncoder(state_dim, vis_d_model)
        self.fusion = nn.Sequential(nn.Linear(vis_d_model * 2, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.control_head = nn.Linear(256, 2)

    def forward(self, img_t2, img_t1, img_t, state_seq, return_attn):
        if return_attn:
            # 当需要注意力图时，显式解包元组
            vis_feat, attn = self.visual_stream(img_t2, img_t1, img_t, return_attn=True)
        else:
            vis_feat = self.visual_stream(img_t2, img_t1, img_t)
        state_feat = self.state_stream(state_seq)
        fused = self.fusion(torch.cat([vis_feat, state_feat], dim=1))
        if return_attn:
            return self.control_head(fused), attn
        return self.control_head(fused)


def create_model():
    return AutoHuberModel()