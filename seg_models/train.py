import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
import segmentation_models_pytorch as smp
from torchmetrics.classification import BinaryJaccardIndex
from PIL import Image
import numpy as np
from tqdm import tqdm

# 自定义数据集
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augmentation=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_names = sorted(os.listdir(img_dir))
        self.mask_names = sorted(os.listdir(mask_dir))
        self.augmentation = augmentation

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)  # 二值化

        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image'].float() / 255.0  # 关键：转为float并归一化
            mask = augmented['mask'].unsqueeze(0)
        else:
            image = ToTensorV2()(image=image)['image'].float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

# albumentations增强
def get_training_augmentation():
    train_transform = [
        albu.Resize(height=512, width=512),  # 这里可以根据你的显存情况调整尺寸
        albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0, value=0),
        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(p=1),
        ToTensorV2(),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        albu.Resize(height=512, width=512),
        albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0, value=0),
        ToTensorV2(),
    ]
    return albu.Compose(test_transform)

# DiceLoss实现
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

def dice_coef(outputs: torch.Tensor, labels: torch.Tensor, threshold=0.5, eps=1e-6):
    outputs = (torch.sigmoid(outputs) > threshold).float()
    labels = labels.float()
    intersection = (outputs * labels).sum()
    union = outputs.sum() + labels.sum()
    dice = (2. * intersection + eps) / (union + eps)
    return dice.item()

# 路径
train_img_dir = '/home/data/sam-unet/shi_ce/xiangdao_data8/Training_Images/'
train_mask_dir = '/home/data/sam-unet/shi_ce/xiangdao_data8/Training_Labels/'
val_img_dir = '/home/data/sam-unet/shi_ce/xiangdao_data8/Validation_Images/'
val_mask_dir = '/home/data/sam-unet/shi_ce/xiangdao_data8/Validation_Labels/'

# 数据集和加载器
train_dataset = SegDataset(train_img_dir, train_mask_dir, augmentation=get_training_augmentation())
val_dataset = SegDataset(val_img_dir, val_mask_dir, augmentation=get_validation_augmentation())
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

# 模型
model = smp.Unet(encoder_name="mobileone_s0", encoder_weights=None, in_channels=3, classes=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 损失和优化器
criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
iou_metric = BinaryJaccardIndex(threshold=0.5).to(device)

# 训练与验证
num_epochs = 100
for epoch in range(num_epochs):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"{'='*50}")

    # 训练阶段
    model.train()
    train_loss = 0
    train_iou = 0
    train_dice = 0
    num_train_batches = 0

    train_bar = tqdm(train_loader, desc='Training', ncols=100, bar_format='{l_bar}{bar:30}{r_bar}', dynamic_ncols=True)
    for images, masks in train_bar:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        # 计算指标
        preds = torch.sigmoid(outputs)
        iou = iou_metric(preds, masks).item()
        dice = dice_coef(outputs, masks)

        train_loss += loss.item()
        train_iou += iou
        train_dice += dice
        num_train_batches += 1

        train_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{iou:.4f}',
            'dice': f'{dice:.4f}'
        })

    avg_train_loss = train_loss / num_train_batches
    avg_train_iou = train_iou / num_train_batches
    avg_train_dice = train_dice / num_train_batches

    # 验证阶段
    model.eval()
    val_loss = 0
    val_iou = 0
    val_dice = 0
    num_val_batches = 0

    val_bar = tqdm(val_loader, desc='Validating', ncols=100, bar_format='{l_bar}{bar:30}{r_bar}', dynamic_ncols=True)
    with torch.no_grad():
        for images, masks in val_bar:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            preds = torch.sigmoid(outputs)
            iou = iou_metric(preds, masks).item()
            dice = dice_coef(outputs, masks)

            val_loss += loss.item()
            val_iou += iou
            val_dice += dice
            num_val_batches += 1

            val_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}',
                'dice': f'{dice:.4f}'
            })

    avg_val_loss = val_loss / num_val_batches
    avg_val_iou = val_iou / num_val_batches
    avg_val_dice = val_dice / num_val_batches

    print(f"\nTrain Loss: {avg_train_loss:.4f} | Train IoU: {avg_train_iou:.4f} | Train Dice: {avg_train_dice:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f} | Val Dice: {avg_val_dice:.4f}")

# 保存模型
os.makedirs("seg_models/pth", exist_ok=True)
torch.save(model.state_dict(), "seg_models/pth/unet_best.pth")