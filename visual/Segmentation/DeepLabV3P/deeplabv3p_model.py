import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# ==================== 1. DeepLabV3+ Architecture ====================

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (空洞空间金字塔池化)"""
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        # 对应 Output Stride = 16 的扩张率
        rates = [1, 6, 12, 18]
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[3], dilation=rates[3], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        # 图像级全局池化
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        
        # 降维融合
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        
    def forward(self, x):
        size = x.shape[-2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=False)
        return self.project(torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1))

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=19, pretrained=True):
        super(DeepLabV3Plus, self).__init__()
        
        # 提取官方 ResNet50 作为 Backbone
        # 修复 torchvision 新版本关于 pretrained 参数弃用的警告
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # ==================== 核心：修改 ResNet50 结构以适应 DeepLab ====================
        # 修改 layer4 使得 Output Stride 变为 16 (而不是默认的 32)
        # 这是为了保留更多的空间分辨率（有利于分割边缘）
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        for i in range(1, 3):
            resnet.layer4[i].conv2.padding = (2, 2)
            resnet.layer4[i].conv2.dilation = (2, 2)
            
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # Low-level features (通道数 256, stride 4)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4  # High-level features (通道数 2048, stride 16)
        
        # 高层语义分支 (ASPP 多尺度捕获)
        self.aspp = ASPP(2048, 256)
        
        # 浅层空间细节分支 (降维)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False), 
            nn.BatchNorm2d(48), 
            nn.ReLU(True)
        )
        
        # 最终的特征融合与解码部分 (Decoder)
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False), 
            nn.BatchNorm2d(256), 
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), 
            nn.BatchNorm2d(256), 
            nn.ReLU(True),
            nn.Conv2d(256, num_classes, 1)
        )
        
    def forward(self, x, size=None):
        h, w = x.shape[-2:]
        
        # 1. Backbone 前向传播
        out = self.stem(x)
        low_level_feat = self.layer1(out)  # 获取浅层特征 -> 提供空间细节 (轮廓、边缘)
        out = self.layer2(low_level_feat)
        out = self.layer3(out)
        high_level_feat = self.layer4(out) # 获取深层特征 -> 提供全局语义意图
        
        # 2. 多尺度上下文提取 (ASPP)
        aspp_out = self.aspp(high_level_feat)
        
        # 3. 浅层特征通道降维 (避免低级特征在拼接时掩盖高级语义)
        low_level_feat = self.low_level_conv(low_level_feat)
        
        # 4. 解码器特征拼接与融合
        aspp_up = F.interpolate(aspp_out, size=low_level_feat.shape[-2:], mode='bilinear', align_corners=False)
        concat = torch.cat([aspp_up, low_level_feat], dim=1)
        out = self.decoder_conv(concat)
        
        # 5. 最后直接使用非线性双线性插值还原至原图尺寸 (输出最终结果)
        target_size = size if size is not None else (h, w)
        return F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)

# ==================== 2. 数据集、损失函数与训练器 (保持与你的一致) ====================

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
    def __init__(self, num_classes=19, ignore_index=255):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        return self.ce(pred, target)

class SegmentationTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, save_dir='checkpoints_deeplabv3p'):
        self.model, self.train_loader, self.val_loader = model, train_loader, val_loader
        self.criterion, self.optimizer, self.scheduler, self.device = criterion, optimizer, scheduler, device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.epoch_loss_history, self.batch_loss_history = [], []
        self.best_miou, self.epoch = 0, 0

    def train_epoch(self):
        self.model.train()
        total_loss, running_loss = 0, 0.0
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
            
            if (batch_idx + 1) % log_interval == 0:
                avg_batch_loss = running_loss / log_interval
                self.batch_loss_history.append({'epoch': self.epoch + 1, 'batch': batch_idx + 1, 'avg_loss': avg_batch_loss})
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
            self.epoch_loss_history.append({'epoch': epoch + 1, 'train_loss': avg_epoch_loss, 'val_loss': val_loss, 'miou': val_miou})
            
            print(f"--- Epoch {epoch+1} Summary ---")
            print(f"Average Train Loss: {avg_epoch_loss:.6f} | Val Loss: {val_loss:.6f} | mIoU: {val_miou:.4f}")
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
                print("保存最佳模型!")

# ==================== 3. 运行的主函数 ====================
def main():
    root_dir = r"E:\Laboratory files\code_project\city_data"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 实例化 DeepLabV3+ 模型
    model = DeepLabV3Plus(num_classes=19, pretrained=True).to(device)

    train_dataset = CityscapesDataset(root_dir, 'train')
    val_dataset = CityscapesDataset(root_dir, 'val')
    print(f"训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")

    # 添加 drop_last=True 以防止最后一个 Batch 大小为 1 时触发 BatchNorm 报错
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 模型权重保存在全新的 checkpoints_deeplabv3p 文件夹下
    trainer = SegmentationTrainer(model, train_loader, val_loader, criterion, optimizer, None, device, save_dir='checkpoints_deeplabv3p')
    trainer.train(num_epochs=50)

if __name__ == "__main__":
    main()
