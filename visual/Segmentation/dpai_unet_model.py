import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo

# ==================== 1. ResNet & DPAI Encoder Components ====================

class Bottleneck(nn.Module):
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

# ==================== 3. Complete DPAI-UNet Model ====================

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

        # DPAI 双向交互模块 (Heavy / Light)
        self.interaction_heavy = AdaptiveBidirectionalInteraction(512, 2048)
        self.interaction_light = AdaptiveBidirectionalInteraction(256, 1024)

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
        x2 = self.layer2(x1)  # 浅层 Fs
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)  # 深层 Fd

        # ------ 2. DPAI 模块 (适应性交互) ------
        x2_enhanced, x4_enhanced = self.interaction_heavy(x2, x4)
        x1_enhanced, x3_enhanced = self.interaction_light(x1, x3)

        # 残差相加，得到融合并增强后的特征
        x1_fused = self.relu(x1 + x1_enhanced)
        x2_fused = self.relu(x2 + x2_enhanced)
        x3_fused = self.relu(x3 + x3_enhanced)
        x4_fused = self.relu(x4 + x4_enhanced)

        # ------ 3. UNet Decoder (特征解码) ------
        # 使用 UNet Decoder 进行逐级上采样并拼接
        features = [x0, x1_fused, x2_fused, x3_fused, x4_fused]
        decoder_out = self.unet_decoder(features)

        # ------ 4. Final Upsampling ------
        # 由于 Decoder 出来的尺寸是原图的 1/2，需要最后插值回原图大小
        target_size = size if size is not None else (x.size(2), x.size(3))
        return F.interpolate(decoder_out, size=target_size, mode='bilinear', align_corners=False)

if __name__ == '__main__':
    # 简单的测试模块 (可选保留)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = DPAI_UNet_Segmentation(num_classes=19, pretrained=False).to(device)
    # dummy_input = torch.randn(2, 3, 512, 1024).to(device)
    # print(f"Input shape: {dummy_input.shape}")
    # out = model(dummy_input)
    # print(f"Output shape: {out.shape}")  # 应该输出 [2, 19, 512, 1024]
    pass

# ==================== 4. 数据处理与损失函数 (从 DPAI2.0.ipynb 拷贝) ====================
import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class CityscapesDataset(Dataset):
    """Cityscapes数据集加载器"""
    CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
               'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
               'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    LABEL_ID_MAPPING = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
                        23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

    def __init__(self, root, split='train', transform=None, target_size=(1024, 512)):
        self.root, self.split, self.transform, self.target_size = root, split, transform, target_size
        self.images = self._get_image_paths()
        self.targets = self._get_target_paths()

    def _get_image_paths(self):
        image_dir = os.path.join(self.root, 'leftImg8bit', self.split)
        images = []
        for city in os.listdir(image_dir):
            city_dir = os.path.join(image_dir, city)
            for img_name in os.listdir(city_dir):
                if img_name.endswith('_leftImg8bit.png'): images.append(os.path.join(city_dir, img_name))
        return sorted(images)

    def _get_target_paths(self):
        target_dir = os.path.join(self.root, 'gtFine', self.split)
        targets = []
        for img_path in self.images:
            img_name = os.path.basename(img_path)
            base_name = img_name.replace('_leftImg8bit.png', '')
            city = os.path.basename(os.path.dirname(img_path))
            target_name = f'{base_name}_gtFine_labelIds.png'
            targets.append(os.path.join(target_dir, city, target_name))
        return sorted(targets)

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        target = Image.open(self.targets[idx])
        orig_size = image.size[::-1]
        image = image.resize(self.target_size, Image.BILINEAR)
        target = target.resize(self.target_size, Image.NEAREST)
        if self.transform: image, target = self.transform(image, target)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        target = self._remap_labels(np.array(target))
        return image, target, orig_size, self.images[idx]

    def _remap_labels(self, target):
        remapped = np.full_like(target, 255)
        for old, new in self.LABEL_ID_MAPPING.items(): remapped[target == old] = new
        return torch.from_numpy(remapped).long()

class CombinedLoss(nn.Module):
    """组合损失"""
    def __init__(self, num_classes=19, ignore_index=255):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.num_classes = num_classes

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        return ce_loss

# ==================== 5. 训练器与主逻辑 ====================

class SegmentationTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, save_dir='checkpoints_unet'):
        self.model, self.train_loader, self.val_loader = model, train_loader, val_loader
        self.criterion, self.optimizer, self.scheduler, self.device = criterion, optimizer, scheduler, device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.train_losses, self.val_losses, self.val_miou_history = [], [], []
        self.epoch_loss_history = []  
        self.batch_loss_history = []
        self.best_miou, self.epoch = 0, 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        running_loss = 0.0
        log_interval = 50

        for batch_idx, (images, targets, _, _) in enumerate(self.train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_val = loss.item()
            total_loss += loss_val
            running_loss += loss_val
            
            # 每 50 个 Batch 记录一次平均 Loss
            if (batch_idx + 1) % log_interval == 0:
                avg_batch_loss = running_loss / log_interval
                self.batch_loss_history.append({
                    'epoch': self.epoch + 1,
                    'batch': batch_idx + 1,
                    'avg_loss': avg_batch_loss
                })
                print(f"  Epoch {self.epoch+1}, Batch {batch_idx+1}, Group Avg Loss: {avg_batch_loss:.4f}")
                running_loss = 0.0
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss, confusion_matrix = 0, np.zeros((19, 19))
        for images, targets, _, _ in self.val_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            outputs = self.model(images)
            total_loss += self.criterion(outputs, targets).item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            for t, p in zip(targets.cpu().numpy(), preds):
                mask = (t != 255)
                label = 19 * t[mask].astype('int') + p[mask]
                confusion_matrix += np.bincount(label, minlength=19**2).reshape(19, 19)
        iu = np.diag(confusion_matrix) / (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - np.diag(confusion_matrix) + 1e-10)
        return total_loss / len(self.val_loader), np.mean(iu)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            avg_epoch_loss = self.train_epoch()
            val_loss, val_miou = self.validate()
            
            self.epoch_loss_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_epoch_loss,
                'val_loss': val_loss,
                'miou': val_miou
            })
            
            print(f"--- Epoch {epoch+1} Summary ---")
            print(f"Average Train Loss: {avg_epoch_loss:.6f}")
            print(f"Average Val Loss:   {val_loss:.6f}")
            print(f"mIoU:               {val_miou:.4f}")
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
                print("保存最佳模型!")

# ==================== 6. 运行脚本 ====================

def main():
    # 路径配置
    root_dir = r"E:\Laboratory files\code_project\city_data"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 实例化改进模型 (DPAI_UNet)
    model = DPAI_UNet_Segmentation(num_classes=19, pretrained=True).to(device)

    train_dataset = CityscapesDataset(root_dir, 'train')
    val_dataset = CityscapesDataset(root_dir, 'val')

    print(f"训练集图片数量: {len(train_dataset)}")
    print(f"验证集图片数量: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    print(f"训练集每个epoch的batch数量: {len(train_loader)}")
    print(f"验证集每个epoch的batch数量: {len(val_loader)}")

    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    trainer = SegmentationTrainer(model, train_loader, val_loader, criterion, optimizer, None, device, save_dir='checkpoints_dpai_unet')
    trainer.train(num_epochs=50)

if __name__ == "__main__":
    main()
