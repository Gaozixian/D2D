import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import os
from PIL import Image
import numpy as np
from datetime import datetime

class DiceLoss(nn.Module):
    """
    Dice Loss: 直接优化模型的 IoU/区域重合度，极大地缓解类别不平衡影响。
    """
    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    def forward(self, pred, target):
        # 对模型的输出进行 Softmax 得到预测概率 (B, C, H, W)
        pred = F.softmax(pred, dim=1)
        
        # 1. 创建 Ignore Mask (去除 Ignore Index 边界，不让其参与计算)
        valid_mask = (target != self.ignore_index)
        target = target * valid_mask.long()
        
        # 2. 将真实的 Target 转换成 One-hot 编码格式 (B, C, H, W)
        target_one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        
        # 3. 把无效的像素点剔除
        target_one_hot = target_one_hot * valid_mask.unsqueeze(1).float()
        pred = pred * valid_mask.unsqueeze(1).float()
        # 4. 计算交集和并集
        intersection = (pred * target_one_hot).sum(dim=(0, 2, 3))  # 交集
        cardinality = pred.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3)) # 并集
        
        # 5. 计算 Dice 系数
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Dice 的计算结果是预测与实际的重合度，因此 Loss 为 (1 - 重合度)
        return 1.0 - dice.mean()
class FocalLoss(nn.Module):
    """
    Focal Loss: 让模型在训练后期专注“难样本”（比如交通信号灯，细小的行人），不再过度更新简单的大背景样本。
    """
    def __init__(self, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    def forward(self, pred, target):
        # 1. 计算原始的交叉熵损失 
        logpt = -self.ce_loss(pred, target)
        
        # 2. 还原出概率值 pt
        pt = torch.exp(logpt)
        
        # 3. 施加 Focal 权重: (1 - pt)^gamma
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        
        # 过滤计算均值
        return focal_loss.mean()
class CombinedLoss(nn.Module):
    """
    高能效组合损失： Focal Loss 捕捉小目标困难样本 + Dice Loss 极大拉高网络对形态(轮廓)的拟合能力。
    这是在 Cityscapes 和医学分割上经常拿第一的标准化组合。
    """
    def __init__(self, num_classes=19, ignore_index=255, focal_w=1.0, dice_w=1.0):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.focal_w = focal_w
        self.dice_w = dice_w
    def forward(self, pred, target):
        f_loss = self.focal(pred, target)
        d_loss = self.dice(pred, target)
        # 你可以根据训练中后期的表现调整权重比例，最经典的起步是 1:1 双管齐下
        total_loss = (self.focal_w * f_loss) + (self.dice_w * d_loss)
        return total_loss

# ==================== 1. 基础组件：Bottleneck ====================
class Bottleneck(nn.Module):
    """ResNet50的瓶颈块 (引用自原文件)"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

# ==================== 2. 核心改进：自适应双向引导交互模块 ====================
class AdaptiveBidirectionalInteraction(nn.Module):
    def __init__(self, s_channels, d_channels, reduction=16):
        super(AdaptiveBidirectionalInteraction, self).__init__()
        self.s_conv7x7 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.s_to_d_mlp = nn.Sequential(
            nn.Linear(1, d_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(d_channels // reduction, d_channels)
        )
        self.d_shared_mlp = nn.Sequential(
            nn.Linear(d_channels, d_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_channels // reduction, d_channels, bias=False)
        )
        mid_channels = 256
        self.d_to_s_conv = nn.Sequential(
            nn.Conv2d(d_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, Fs, Fd):
        avg_p = F.adaptive_avg_pool2d(Fd, 1).view(Fd.size(0), -1)
        max_p = F.adaptive_max_pool2d(Fd, 1).view(Fd.size(0), -1)
        d_stats = self.d_shared_mlp(avg_p) + self.d_shared_mlp(max_p)
        Ac = self.sigmoid(d_stats)

        s_avg = torch.mean(Fs, dim=1, keepdim=True)
        s_max, _ = torch.max(Fs, dim=1, keepdim=True)
        s_desc = torch.cat([s_avg, s_max], dim=1)

        Gds = self.d_to_s_conv(Fd * Ac.view(Fd.size(0), Fd.size(1), 1, 1))
        Gds_up = F.interpolate(Gds, size=(Fs.size(2), Fs.size(3)), mode='bilinear', align_corners=False)

        s_desc_fused = s_desc * Gds_up
        As = self.sigmoid(self.s_conv7x7(s_desc_fused))
        Fs_out = Fs * As

        s_guide_vec = F.adaptive_avg_pool2d(As, 1).view(As.size(0), -1)
        Gsd = self.s_to_d_mlp(s_guide_vec)

        final_Ac = self.sigmoid(d_stats + Gsd).view(Fd.size(0), Fd.size(1), 1, 1)
        Fd_out = Fd * final_Ac

        return Fs_out, Fd_out
# ==================== CBAM注意力模块====================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class CBAM(nn.Module):
    def __init__(self, gate_channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelAttention(gate_channels, ratio)
        self.SpatialGate = SpatialAttention(kernel_size)

    def forward(self, x):
        x_out = x * self.ChannelGate(x)
        x_out = x_out * self.SpatialGate(x_out)
        return x_out

# ==================== 2. UNet Decoder Components ====================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
    # 前向传播函数，输入为x
    # self.double_conv表示一个由两个卷积层组成的序列
    # 将输入x传递给double_conv模块进行处理
        return self.double_conv(x)
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels=[64, 256, 512, 1024, 2048],
                 decoder_channels=[256, 128, 64, 32], n_classes=19, bilinear=True):
        super(UNetDecoder, self).__init__()
        enc_chs = encoder_channels[::-1]
        self.up1 = UpBlock(enc_chs[0] + enc_chs[1], decoder_channels[0], bilinear)
        self.up2 = UpBlock(decoder_channels[0] + enc_chs[2], decoder_channels[1], bilinear)
        self.up3 = UpBlock(decoder_channels[1] + enc_chs[3], decoder_channels[2], bilinear)
        self.up4 = UpBlock(decoder_channels[2] + enc_chs[4], decoder_channels[3], bilinear)
        self.outc = nn.Conv2d(decoder_channels[3], n_classes, kernel_size=1)

    def forward(self, features):
        x0, x1, x2, x3, x4 = features
        d4 = self.up1(x4, x3)
        d3 = self.up2(d4, x2)
        d2 = self.up3(d3, x1)
        d1 = self.up4(d2, x0)
        return self.outc(d1)
# ==================== 3. DPAI_Unet ====================
class DPAI_UNet_Segmentation(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=19, pretrained=False):
        super(DPAI_UNet_Segmentation, self).__init__()
        self.inplanes = 64

        # Encoder 前缀
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ResNet 层级
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)
        # DPAI 双向交互模块 (Heavy / Light)
        self.interaction_heavy = AdaptiveBidirectionalInteraction(512, 2048)
        self.interaction_light = AdaptiveBidirectionalInteraction(256, 1024)
        self.x0_enhance = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64)
        )
        # ==============================================================================

        # U-Net Decoder (融合深浅层特征)
        self.unet_decoder = UNetDecoder(
            encoder_channels=[64, 256, 512, 1024, 2048],
            decoder_channels=[256, 128, 64, 32],
            n_classes=num_classes
        )

        self._initialize_weights()
        if pretrained:
            self._load_pretrained_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()

    def _load_pretrained_weights(self):
        try:
            pre_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
            model_dict = self.state_dict()
            pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pre_dict)
            self.load_state_dict(model_dict)
            print("Successfully loaded pre-trained ResNet-50 weights in Encoder!")
        except Exception as e:
            print("Failed to load pre-trained weights:", e)

    def forward(self, x, size=None):
        # ------ 1. Encoder (特征提取) ------
        x0 = self.relu(self.bn1(self.conv1(x)))
        x_low = self.maxpool(x0)

        x1 = self.layer1(x_low)
        x1 = self.cbam1(x1)
        x2 = self.layer2(x1)  # 浅层 Fs
        x2 = self.cbam2(x2)
        x3 = self.layer3(x2)
        x3 = self.cbam3(x3)
        x4 = self.layer4(x3)  # 深层 Fd
        x4 = self.cbam4(x4)

        # ------ 2. DPAI 模块 (适应性交互) ------
        x0_processed = self.x0_enhance(x0)
        x2_enhanced, x4_enhanced = self.interaction_heavy(x2, x4)
        x1_enhanced, x3_enhanced = self.interaction_light(x1, x3)

        # 残差相加，得到融合并增强后的特征
        x0_fused = self.relu(x0 + x0_processed)  # 经典的 ResNet 残差处理！
        x1_fused = self.relu(x1 + x1_enhanced)
        x2_fused = self.relu(x2 + x2_enhanced)
        x3_fused = self.relu(x3 + x3_enhanced)
        x4_fused = self.relu(x4 + x4_enhanced)

        # ------ 3. UNet Decoder (特征解码) ------
        # 使用 UNet Decoder 进行逐级上采样并拼接
        features = [x0_fused, x1_fused, x2_fused, x3_fused, x4_fused]
        decoder_out = self.unet_decoder(features)

        # ------ 4. Final Upsampling ------
        # 由于 Decoder 出来的尺寸是原图的 1/2，需要最后插值回原图大小
        target_size = size if size is not None else (x.size(2), x.size(3))
        return F.interpolate(decoder_out, size=target_size, mode='bilinear', align_corners=False)


def load_unet_model_weight(model, weight_path, device):
    """加载之前跑出来的最好的 .pth 权重文件"""
    if not os.path.exists(weight_path):
        print(f"❌ 找不到权重文件: {weight_path}")
        return model

    # 1. 挂载权重
    state_dict = torch.load(weight_path, map_location=device)

    # 2. 清理多显卡训练时可能残存的并行 'module.' 前缀 (剥离操作)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()  # 设置为推理模式至关重要（关闭 Dropout 和 BN 特性）
    print("✅ 预训练权重装填成功！")
    return model

def predict_and_visualize(model, image_path, device, target_size=(1024, 512), save_dir='visual_results_unet'):
    """接受一张图片路径，送入 UNet 发光并画图"""

    # Cityscapes 的原生标注字典
    class_colors = {
        0: [128, 64, 128], 1: [244, 35, 232], 2: [70, 70, 70], 3: [102, 102, 156],
        4: [190, 153, 153], 5: [153, 153, 153], 6: [250, 170, 30], 7: [220, 220, 0],
        8: [107, 142, 35], 9: [152, 251, 152], 10: [70, 130, 180], 11: [220, 20, 60],
        12: [255, 0, 0], 13: [0, 0, 142], 14: [0, 0, 70], 15: [0, 60, 100],
        16: [0, 80, 100], 17: [0, 0, 230], 18: [119, 11, 32], 255: [0, 0, 0]
    }

    os.makedirs(save_dir, exist_ok=True)

    # 1. 完全复刻原装 Loader 的数据图片处理
    original_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = original_img.size # 记录原图长宽，推断完了要等比例放大回去

    # Resize -> 维度置换 -> 转 Float 再化至 [0, 1] 空间 (PIL 宽度在左，高度在右)
    resized_img = original_img.resize(target_size, Image.BILINEAR)
    img_tensor = torch.from_numpy(np.array(resized_img)).permute(2, 0, 1).float() / 255.0
    input_tensor = img_tensor.unsqueeze(0).to(device)

    # 2. 无梯度通过网络，推理出特征
    with torch.no_grad():
        output = model(input_tensor)
        # 把预测特征图暴降拉回真实城市街道拍摄照片的高宽（通常是 2048 x 1024）
        output = F.interpolate(output, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

    # 张量取所有类别的 Argmax 作为最大确信的标签索引 (H, W)
    pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 3. 张量画布上色
    color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    for label, color in class_colors.items():
        color_mask[pred_mask == label] = color

    color_mask_img = Image.fromarray(color_mask)
