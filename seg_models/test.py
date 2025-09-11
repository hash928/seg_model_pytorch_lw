import os
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
import torch.nn as nn
from torchmetrics.classification import BinaryJaccardIndex

# DiceLoss实现 (与训练脚本保持一致)
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

def get_test_augmentation():
    """测试时使用的数据增强，主要进行尺寸统一。"""
    test_transform = [
        albu.Resize(height=512, width=512),
        albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0, value=0),
        ToTensorV2(),
    ]
    return albu.Compose(test_transform)

class TestDataset(Dataset):
    """用于测试图像的自定义数据集。"""
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
        original_height, original_width = image.shape[:2] # 记录原始尺寸
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)  # 二值化

        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)
        else:
            image = ToTensorV2()(image=image)['image']
            mask = torch.from_numpy(mask).unsqueeze(0)

        image = image.float() / 255.0
        return image, mask, self.img_names[idx], (original_height, original_width)

def test():
    """执行模型测试。"""
    # ----- 用户配置 -----
    test_img_dir = '/home/data/sam-unet/shi_ce/xiangdao_data8/Test_Images/'
    test_mask_dir = '/home/data/sam-unet/shi_ce/xiangdao_data8/Test_Labels/' 
    model_path = "seg_models/pth/unet_best.pth"
    output_dir = "seg_models/test_output/"
    # -------------------

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = smp.Unet(encoder_name="mobileone_s0", encoder_weights=None, activation="sigmoid", in_channels=3, classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # 损失函数和评估指标
    criterion = DiceLoss()
    iou_metric = BinaryJaccardIndex(threshold=0.5).to(device)

    # 数据集和加载器
    test_dataset = TestDataset(test_img_dir, test_mask_dir, augmentation=get_test_augmentation())
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    test_loss = 0
    test_iou = 0
    test_dice = 0
    num_test_batches = 0

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='正在测试', ncols=100)
        for images, masks, img_names, original_sizes in test_bar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            # 计算指标 (在512x512上计算，与训练保持一致)
            preds = torch.sigmoid(outputs)
            iou = iou_metric(preds, masks).item()
            dice = dice_coef(outputs, masks)

            test_loss += loss.item()
            test_iou += iou
            test_dice += dice
            num_test_batches += 1

            # 保存预测结果图像
            for i in range(preds.size(0)):
                # 获取512x512的预测掩码
                pred_mask_512 = (preds[i].squeeze().cpu().numpy() > 0.5).astype(np.uint8)
                
                # 获取原始尺寸
                original_h, original_w = original_sizes[0][i].item(), original_sizes[1][i].item()
                
                # 将掩码缩放回原始尺寸 (cv2.resize需要 (宽, 高) 的顺序)
                pred_mask_resized = cv2.resize(pred_mask_512, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

                pred_mask_resized = pred_mask_resized * 255
                output_path = os.path.join(output_dir, img_names[i])
                cv2.imwrite(output_path, pred_mask_resized)

            test_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}',
                'dice': f'{dice:.4f}'
            })

    avg_test_loss = test_loss / num_test_batches
    avg_test_iou = test_iou / num_test_batches
    avg_test_dice = test_dice / num_test_batches

    print(f"\n{'='*20} 测试结果 {'='*20}")
    print(f"平均交并比 (Avg IoU): {avg_test_iou:.4f}")
    print(f"平均Dice系数 (Avg Dice): {avg_test_dice:.4f}")
    print(f"{'='*51}")
    print(f"\n测试完成。预测结果已保存至: {output_dir}")

if __name__ == "__main__":
    test()
