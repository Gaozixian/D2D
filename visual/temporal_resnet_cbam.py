# -*- coding: utf-8 -*-
"""
时序ResNet_CBAM + Transformer融合网络
将三张时序图像（t-2, t-1, t）通过ResNet_CBAM处理后，将t-1和t-2时刻的特征放入K、V，t时刻的特征放入Q
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from pathlib import Path

# ==================== 原始ResNet_CBAM代码（复用） ====================

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Bottleneck(nn.Module):
    """ResNet50的瓶颈块，集成CBAM注意力机制"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        
        # CBAM注意力模块
        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # 应用CBAM注意力机制
        out = self.ca(out) * out
        out = self.sa(out) * out
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """完整的ResNet网络结构，集成CBAM注意力机制"""

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet的各个层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 全局平均池化（不展平，保留空间结构）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """构建ResNet的层级"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def get_feature_maps(self, x):
        """获取特征图（不进行全局池化）"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x  # 返回 (batch_size, channels, height, width)

def create_resnet_cbam():
    """创建ResNet50+CBAM模型"""
    return ResNet(Bottleneck, [3, 4, 6, 3])

# ==================== 时序ResNet_CBAM网络 ====================

class TemporalResNetCBAM(nn.Module):
    """时序ResNet_CBAM网络 - 处理三张时序图像"""
    
    def __init__(self, 
                 use_shared_weights=True, # 共享权重的参数
                 d_model=512,
                 num_heads=8,
                 num_layers=6,
                 image_size=224):
        super(TemporalResNetCBAM, self).__init__()
        
        self.use_shared_weights = use_shared_weights
        self.image_size = image_size
        
        # 计算特征图尺寸（假设输入224x224）
        # 224 -> 112 -> 56 -> 28 -> 14 -> 7 (经过5次下采样)
        self.feature_map_size = image_size // (2 ** 5)  # 224 -> 7
        self.num_patches = self.feature_map_size ** 2  # 49个patches
        
        # 1. 创建ResNet_CBAM网络
        if use_shared_weights:
            # 权重共享：使用同一个ResNet处理三张图像
            self.resnet = create_resnet_cbam()
            print("使用权重共享的ResNet_CBAM")
        else:
            # 权重不共享：创建三个独立的ResNet
            self.resnet_t_minus_2 = create_resnet_cbam()  # t-2时刻
            self.resnet_t_minus_1 = create_resnet_cbam()  # t-1时刻
            self.resnet_t = create_resnet_cbam()          # t时刻
            print("使用独立权重的ResNet_CBAM")
        
        # 2. 特征投影层
        self.feature_projection = nn.Linear(2048, d_model)
        
        # 3. 时序位置编码
        self.temporal_pos_encoding = nn.Parameter(
            torch.randn(3, d_model)
        )  # 为三个时间步添加位置信息
        
        # 4. 多头交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 5. Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 6. 输出层
        self.output_projection = nn.Linear(d_model, 1000)  # 假设1000类分类
        
    def forward(self, img_t_minus_2, img_t_minus_1, img_t):
        """
        Args:
            img_t_minus_2: t-2时刻图像, (batch_size, 3, H, W)
            img_t_minus_1: t-1时刻图像, (batch_size, 3, H, W)  
            img_t: t时刻图像, (batch_size, 3, H, W)
            
        Returns:
            output: 最终输出
            debug_info: 调试信息
        """
        batch_size = img_t.size(0)
        
        # 步骤1: 使用ResNet_CBAM提取三张图像的特征
        if self.use_shared_weights:
            # 权重共享
            feat_t_minus_2 = self.resnet.get_feature_maps(img_t_minus_2)
            feat_t_minus_1 = self.resnet.get_feature_maps(img_t_minus_1)
            feat_t = self.resnet.get_feature_maps(img_t)
        else:
            # 权重不共享
            feat_t_minus_2 = self.resnet_t_minus_2.get_feature_maps(img_t_minus_2)
            feat_t_minus_1 = self.resnet_t_minus_1.get_feature_maps(img_t_minus_1)
            feat_t = self.resnet_t.get_feature_maps(img_t)
        
        # 步骤2: 将特征图展平为序列
        def flatten_features(features):
            batch_size, channels, height, width = features.shape
            return features.view(batch_size, channels, -1).transpose(1, 2)
        
        feat_flat_t_minus_2 = flatten_features(feat_t_minus_2)  # (batch_size, 49, 2048)
        feat_flat_t_minus_1 = flatten_features(feat_t_minus_1)  # (batch_size, 49, 2048)
        feat_flat_t = flatten_features(feat_t)                  # (batch_size, 49, 2048)
        
        # 步骤3: 特征投影到Transformer维度
        proj_t_minus_2 = self.feature_projection(feat_flat_t_minus_2)  # (batch_size, 49, d_model)
        proj_t_minus_1 = self.feature_projection(feat_flat_t_minus_1)  # (batch_size, 49, d_model)
        proj_t = self.feature_projection(feat_flat_t)                  # (batch_size, 49, d_model)
        
        # 步骤4: 添加时序位置编码
        # 为每个时间步添加不同的位置编码
        temporal_pos = self.temporal_pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        # temporal_pos: (batch_size, 3, d_model)
        
        # 将三个时刻的特征与位置编码结合
        seq_features = torch.stack([proj_t_minus_2, proj_t_minus_1, proj_t], dim=1)
        # seq_features: (batch_size, 3, 49, d_model)
        
        # 重塑以添加位置编码
        batch_size, num_timesteps, seq_len, d_model = seq_features.shape
        seq_features_flat = seq_features.view(batch_size, num_timesteps * seq_len, d_model)
        temporal_pos_flat = temporal_pos.view(batch_size, num_timesteps, 1, d_model).expand(-1, -1, seq_len, -1)
        temporal_pos_flat = temporal_pos_flat.view(batch_size, num_timesteps * seq_len, d_model)
        
        features_with_pos = seq_features_flat + temporal_pos_flat
        
        # 步骤5: 生成K, V, Q
        # K, V: t-2和t-1时刻的特征
        kv_features = torch.cat([proj_t_minus_2, proj_t_minus_1], dim=1)
        # kv_features: (batch_size, 98, d_model)
        
        # Q: t时刻的特征
        q_features = proj_t  # (batch_size, 49, d_model)
        
        # 步骤6: 交叉注意力机制
        # Q与K,V进行交叉注意力
        attended_output, attention_weights = self.cross_attention(
            query=q_features,
            key=kv_features,
            value=kv_features
        )
        # attended_output: (batch_size, 49, d_model)
        
        # 步骤7: Transformer编码器处理
        # 将当前时刻的输出与所有时刻的特征一起处理
        enhanced_features = torch.cat([attended_output, features_with_pos], dim=1)
        encoder_output = self.transformer_encoder(enhanced_features)
        
        # 取当前时刻对应的输出
        current_output = encoder_output[:, :49, :]  # (batch_size, 49, d_model)
        
        # 步骤8: 全局池化和分类
        final_output = current_output.mean(dim=1)  # (batch_size, d_model)
        final_output = self.output_projection(final_output)  # (batch_size, 1000)
        
        # 返回结果和调试信息
        debug_info = {
            'feat_t_minus_2': feat_t_minus_2,
            'feat_t_minus_1': feat_t_minus_1,
            'feat_t': feat_t,
            'feat_flat_t_minus_2': feat_flat_t_minus_2,
            'feat_flat_t_minus_1': feat_flat_t_minus_1,
            'feat_flat_t': feat_flat_t,
            'proj_t_minus_2': proj_t_minus_2,
            'proj_t_minus_1': proj_t_minus_1,
            'proj_t': proj_t,
            'kv_features': kv_features,
            'q_features': q_features,
            'attention_weights': attention_weights,
            'attended_output': attended_output,
            'encoder_output': encoder_output,
            'current_output': current_output
        }
        
        return final_output, debug_info

# ==================== 简化版本（更直接的实现） ====================

class SimpleTemporalResNetCBAM(nn.Module):
    """简化版时序ResNet_CBAM - 更直接的处理方式"""
    
    def __init__(self, 
                 use_shared_weights=True,
                 d_model=512,
                 num_heads=8):
        super(SimpleTemporalResNetCBAM, self).__init__()
        
        self.use_shared_weights = use_shared_weights
        self.d_model = d_model
        
        # 创建ResNet_CBAM网络
        if use_shared_weights:
            self.resnet = create_resnet_cbam()
        else:
            self.resnet_t_minus_2 = create_resnet_cbam()
            self.resnet_t_minus_1 = create_resnet_cbam()
            self.resnet_t = create_resnet_cbam()
        
        # Q, K, V投影层
        self.q_proj = nn.Linear(2048, d_model)
        self.k_proj = nn.Linear(2048, d_model)
        self.v_proj = nn.Linear(2048, d_model)
        
        # 多头注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 输出层
        self.output_proj = nn.Linear(d_model, 1000)
        
    def forward(self, img_t_minus_2, img_t_minus_1, img_t):
        """简化版前向传播"""
        batch_size = img_t.size(0)
        
        # 提取特征
        if self.use_shared_weights:
            feat_t_minus_2 = self.resnet.get_feature_maps(img_t_minus_2)
            feat_t_minus_1 = self.resnet.get_feature_maps(img_t_minus_1)
            feat_t = self.resnet.get_feature_maps(img_t)
        else:
            feat_t_minus_2 = self.resnet_t_minus_2.get_feature_maps(img_t_minus_2)
            feat_t_minus_1 = self.resnet_t_minus_1.get_feature_maps(img_t_minus_1)
            feat_t = self.resnet_t.get_feature_maps(img_t)
        
        # 展平特征
        def flatten_features(features):
            return features.view(features.size(0), features.size(1), -1).transpose(1, 2)
        
        feat_flat_t_minus_2 = flatten_features(feat_t_minus_2)
        feat_flat_t_minus_1 = flatten_features(feat_t_minus_1)
        feat_flat_t = flatten_features(feat_t)
        
        # 生成K, V, Q
        # K, V: t-2和t-1时刻的特征
        k_features = torch.cat([feat_flat_t_minus_2, feat_flat_t_minus_1], dim=1)
        v_features = k_features  # K和V使用相同的特征
        
        # Q: t时刻的特征
        q_features = feat_flat_t
        
        # 投影到统一维度
        K = self.k_proj(k_features)  # (batch_size, 98, d_model)
        V = self.v_proj(v_features)  # (batch_size, 98, d_model)
        Q = self.q_proj(q_features)  # (batch_size, 49, d_model)
        
        # 交叉注意力
        attended_output, attention_weights = self.attention(Q, K, V)
        
        # 全局池化和分类
        final_output = attended_output.mean(dim=1)  # (batch_size, d_model)
        final_output = self.output_proj(final_output)  # (batch_size, 1000)
        
        return final_output, {
            'Q': Q, 'K': K, 'V': V,
            'attention_weights': attention_weights,
            'attended_output': attended_output
        }

# ==================== 位置编码增强版本 ====================

class TemporalPositionalEncoding(nn.Module):
    """时序位置编码"""
    def __init__(self, d_model, max_len=3):  # 3个时间步
        super(TemporalPositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, d_model = x.shape
        # 重复位置编码以匹配序列长度
        pe_expanded = self.pe.unsqueeze(0).repeat(batch_size, 1, 1)
        return x + pe_expanded

class EnhancedTemporalResNetCBAM(nn.Module):
    """增强版时序ResNet_CBAM - 带有时序位置编码"""
    
    def __init__(self, use_shared_weights=True, d_model=512, num_heads=8):
        super(EnhancedTemporalResNetCBAM, self).__init__()
        
        self.use_shared_weights = use_shared_weights
        
        # 创建ResNet_CBAM网络
        if use_shared_weights:
            self.resnet = create_resnet_cbam()
        else:
            self.resnet_t_minus_2 = create_resnet_cbam()
            self.resnet_t_minus_1 = create_resnet_cbam()
            self.resnet_t = create_resnet_cbam()
        
        # 特征投影层
        self.feature_proj = nn.Linear(2048, d_model)
        
        # 时序位置编码
        self.temporal_pos_encoding = TemporalPositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 输出层
        self.output_proj = nn.Linear(d_model, 1000)
        
    def forward(self, img_t_minus_2, img_t_minus_1, img_t):
        batch_size = img_t.size(0)
        
        # 提取特征
        if self.use_shared_weights:
            feat_t_minus_2 = self.resnet.get_feature_maps(img_t_minus_2)
            feat_t_minus_1 = self.resnet.get_feature_maps(img_t_minus_1)
            feat_t = self.resnet.get_feature_maps(img_t)
        else:
            feat_t_minus_2 = self.resnet_t_minus_2.get_feature_maps(img_t_minus_2)
            feat_t_minus_1 = self.resnet_t_minus_1.get_feature_maps(img_t_minus_1)
            feat_t = self.resnet_t.get_feature_maps(img_t)
        
        # 展平特征
        def flatten_features(features):
            return features.view(features.size(0), features.size(1), -1).transpose(1, 2)
        
        feat_flat_t_minus_2 = flatten_features(feat_t_minus_2)
        feat_flat_t_minus_1 = flatten_features(feat_t_minus_1)
        feat_flat_t = flatten_features(feat_t)
        
        # 投影到统一维度
        proj_t_minus_2 = self.feature_proj(feat_flat_t_minus_2)
        proj_t_minus_1 = self.feature_proj(feat_flat_t_minus_1)
        proj_t = self.feature_proj(feat_flat_t)
        
        # 构建序列：t-2, t-1, t
        sequence = torch.stack([proj_t_minus_2, proj_t_minus_1, proj_t], dim=1)
        # sequence: (batch_size, 3, 49, d_model)
        
        # 重塑并添加位置编码
        batch_size, num_timesteps, seq_len, d_model = sequence.shape
        sequence_flat = sequence.view(batch_size, num_timesteps * seq_len, d_model)
        
        # 添加位置编码
        sequence_with_pos = self.temporal_pos_encoding(sequence_flat)
        
        # Transformer处理
        transformer_output = self.transformer(sequence_with_pos)
        
        # 取当前时刻（t）的输出
        current_output = transformer_output[:, 2*seq_len:3*seq_len, :]  # t时刻对应索引98:147
        
        # 全局池化和分类
        final_output = current_output.mean(dim=1)
        final_output = self.output_proj(final_output)
        
        return final_output, {
            'sequence': sequence,
            'sequence_with_pos': sequence_with_pos,
            'transformer_output': transformer_output,
            'current_output': current_output
        }

# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("=== 时序ResNet_CBAM网络演示 ===\n")
    
    # 创建测试数据 - 三张时序图像
    batch_size = 2
    image_size = 224
    
    # t-2时刻图像
    img_t_minus_2 = torch.randn(batch_size, 3, image_size, image_size)
    # t-1时刻图像  
    img_t_minus_1 = torch.randn(batch_size, 3, image_size, image_size)
    # t时刻图像
    img_t = torch.randn(batch_size, 3, image_size, image_size)
    
    print(f"输入图像尺寸:")
    print(f"  t-2时刻: {img_t_minus_2.shape}")
    print(f"  t-1时刻: {img_t_minus_1.shape}")
    print(f"  t时刻: {img_t.shape}")
    
    # 测试权重共享版本
    print("\n--- 权重共享版本 ---")
    model_shared = TemporalResNetCBAM(use_shared_weights=True)
    
    with torch.no_grad():
        output_shared, debug_shared = model_shared(img_t_minus_2, img_t_minus_1, img_t)
    
    print(f"输出尺寸: {output_shared.shape}")
    print(f"K特征尺寸: {debug_shared['kv_features'].shape}")
    print(f"Q特征尺寸: {debug_shared['q_features'].shape}")
    print(f"注意力权重尺寸: {debug_shared['attention_weights'].shape}")
    
    # 测试权重不共享版本
    print("\n--- 权重不共享版本 ---")
    model_independent = TemporalResNetCBAM(use_shared_weights=False)
    
    with torch.no_grad():
        output_independent, debug_independent = model_independent(img_t_minus_2, img_t_minus_1, img_t)
    
    print(f"输出尺寸: {output_independent.shape}")
    
    # 测试简化版本
    print("\n--- 简化版本 ---")
    model_simple = SimpleTemporalResNetCBAM(use_shared_weights=True)
    
    with torch.no_grad():
        output_simple, debug_simple = model_simple(img_t_minus_2, img_t_minus_1, img_t)
    
    print(f"输出尺寸: {output_simple.shape}")
    print(f"Q尺寸: {debug_simple['Q'].shape}")
    print(f"K尺寸: {debug_simple['K'].shape}")
    print(f"V尺寸: {debug_simple['V'].shape}")
    
    # 测试增强版本
    print("\n--- 增强版本（带位置编码）---")
    model_enhanced = EnhancedTemporalResNetCBAM(use_shared_weights=True)
    
    with torch.no_grad():
        output_enhanced, debug_enhanced = model_enhanced(img_t_minus_2, img_t_minus_1, img_t)
    
    print(f"输出尺寸: {output_enhanced.shape}")
    
    # 参数数量统计
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== 参数数量统计 ===")
    print(f"权重共享版本: {count_parameters(model_shared):,}")
    print(f"权重不共享版本: {count_parameters(model_independent):,}")
    print(f"简化版本: {count_parameters(model_simple):,}")
    print(f"增强版本: {count_parameters(model_enhanced):,}")
    
    print(f"\n=== 实现说明 ===")
    print("1. 权重共享版本：使用同一个ResNet处理三张时序图像，参数效率高")
    print("2. 权重不共享版本：三个独立的ResNet，特征提取能力更强")
    print("3. 简化版本：更直接的处理方式，代码简洁")
    print("4. 增强版本：添加了时序位置编码，更好地建模时序关系")
    print("\n核心思想：")
    print("- K, V: t-2和t-1时刻的图像特征")
    print("- Q: t时刻的图像特征")
    print("- 通过交叉注意力让当前时刻关注历史时刻的信息")