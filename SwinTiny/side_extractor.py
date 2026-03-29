import torch
import torch.nn as nn
import torch.nn.functional as F

# 从你新整合好的 seg_model 中引入基础模块
from seg_model import Bottleneck, AdaptiveBidirectionalInteraction

class ResNet_Backbone_4CH(nn.Module):
    """专门为接收 4 通道 (RGB + Mask) 修改的 ResNet50 骨干网络"""
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet_Backbone_4CH, self).__init__()

        # 🌟 关键修改: in_channels 从 3 改为 4
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)  # Stage 2: 512 通道 (浅层细节 Fs)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)  # Stage 4: 2048 通道 (深层语义 Fd)
        return x2, x4

class SideViewDPAIExtractor(nn.Module):
    """侧向专属视角提取器，使用 DPAI 进行深浅层交互，并切片为序列"""
    def __init__(self, out_dim=384):
        super().__init__()
        self.backbone = ResNet_Backbone_4CH(Bottleneck, [3, 4, 6, 3])
        self.interaction = AdaptiveBidirectionalInteraction(s_channels=512, d_channels=2048)
        self.norm_layer = nn.GroupNorm(32, 2048)
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, out_dim, kernel_size=1)
        )

    def load_pretrained_resnet(self):
        import torchvision.models as models
        print(">>> 正在加载侧向视角 ResNet50 预训练权重...")
        official_model = models.resnet50(pretrained=True)
        official_dict = official_model.state_dict()
        model_dict = self.backbone.state_dict()

        conv1_weight = official_dict['conv1.weight']
        new_conv1_weight = torch.zeros(64, 4, 7, 7)
        new_conv1_weight[:, :3, :, :] = conv1_weight
        new_conv1_weight[:, 3, :, :] = conv1_weight.mean(dim=1)
        official_dict['conv1.weight'] = new_conv1_weight

        model_dict.update({k: v for k, v in official_dict.items() if k in model_dict and v.size() == model_dict[k].size()})
        self.backbone.load_state_dict(model_dict)
        print(">>> 侧向视角权重适配完成！")

    def forward(self, rgb, mask):
        x = torch.cat([rgb, mask.float()], dim=1)
        fs, fd = self.backbone(x)
        
        fs_enhanced, fd_enhanced = self.interaction(fs, fd)
        combined_fd = self.norm_layer(fd + fd_enhanced)
        
        feat = self.fusion_conv(combined_fd)  # [B, 384, H', W']
        feat_seq = feat.flatten(2).transpose(1, 2) # [B, 49, 384]
        return feat_seq
