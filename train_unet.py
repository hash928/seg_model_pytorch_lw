#!/usr/bin/env python3
# Train UNet on LabPics 1 dataset
# 基于SAM2训练代码改编的UNet训练脚本

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import csv
from datetime import datetime
import random
import torch.optim as opt
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
import re
import sys
from torch.utils.data import DataLoader

# 导入UNet模型
from models.UNetb import unet_base, unet_resnet

# 导入数据集
from dataset import FullDataset

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_shell_args(shell_file='train_unet.sh'):
    """从shell脚本中解析参数"""
    args_dict = {}
    try:
        with open(shell_file, 'r') as f:
            content = f.read()
            
        # 移除注释行
        lines = content.split('\n')
        active_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                active_lines.append(line)
        content = '\n'.join(active_lines)
            
        # 使用正则表达式匹配参数
        pattern = r'--(\w+)\s+"([^"]+)"'
        matches = re.findall(pattern, content)
        
        for key, value in matches:
            # 转换数值类型
            if value.isdigit():
                value = int(value)
            elif re.match(r'^-?\d*\.\d+$', value):
                value = float(value)
            args_dict[key] = value
            
        # 处理布尔参数
        if '--freeze_backbone' in content and not any(line.strip().startswith('#') and '--freeze_backbone' in line for line in lines):
            args_dict['freeze_backbone'] = True
            
        print("从shell脚本读取的参数：")
        for key, value in args_dict.items():
            print(f"{key}: {value}")
            
    except FileNotFoundError:
        print(f"警告：未找到shell脚本 {shell_file}，将使用命令行参数")
    except Exception as e:
        print(f"警告：解析shell脚本时出错：{e}，将使用命令行参数")
        
    return args_dict

# 创建参数解析器
parser = argparse.ArgumentParser("UNet Model Training")
parser.add_argument("--model_type", type=str, default="unet_base", 
                    choices=["unet_base", "unet_resnet18", "unet_resnet34", "unet_resnet50", "unet_resnet101", "unet_resnet152"],
                    help="UNet模型类型")
parser.add_argument("--pretrained", action="store_true", 
                    help="是否使用预训练的ResNet backbone")
parser.add_argument("--train_image_path", type=str, required=True, 
                    help="训练图像路径")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="训练掩码路径")
parser.add_argument("--val_image_path", type=str, required=True, 
                    help="验证图像路径")
parser.add_argument("--val_mask_path", type=str, required=True,
                    help="验证掩码路径")
parser.add_argument('--save_path', type=str, required=True,
                    help="模型保存路径")
parser.add_argument("--epoch", type=int, default=200, 
                    help="训练轮数")
parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--freeze_backbone", action="store_true",
                    help="是否冻结主干网络参数")
parser.add_argument("--num_workers", type=int, default=4,
                    help="数据加载时使用的子进程数量")
parser.add_argument("--input_size", type=int, default=352,
                    help="输入图像尺寸")
parser.add_argument("--num_classes", type=int, default=1,
                    help="分割类别数（1为二值分割）")

# 首先尝试从shell脚本读取参数
shell_args = parse_shell_args()

# 将shell脚本中的参数转换为命令行参数格式
if shell_args:
    sys.argv = [sys.argv[0]]  # 清空现有参数
    for key, value in shell_args.items():
        if isinstance(value, bool):
            if value:
                sys.argv.append(f"--{key}")
        else:
            sys.argv.extend([f"--{key}", str(value)])

# 解析命令行参数
args = parser.parse_args()

def dice_loss(pred, target, smooth=1e-6):
    """Dice损失函数"""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def bce_dice_loss(pred, target, bce_weight=0.5, dice_weight=0.5):
    """BCE + Dice组合损失函数"""
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce_weight * bce + dice_weight * dice

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal损失函数，用于处理类别不平衡"""
    pred = torch.sigmoid(pred)
    ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
    p_t = pred * target + (1 - pred) * (1 - target)
    loss = ce_loss * ((1 - p_t) ** gamma)
    
    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    
    return loss.mean()

def structure_loss(pred, mask):
    """结构损失函数，兼容3D/4D输入"""
    # 保证输入为(N,1,H,W)
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def calculate_metrics(pred, mask, threshold=0.5):
    """计算IoU和Dice指标"""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * mask).sum()
    union = (pred + mask).sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred.sum() + mask.sum() + 1e-6)
    return iou.item(), dice.item()

def print_model_structure(model):
    """打印模型结构"""
    print(f"\n{'='*50}")
    print(f"模型类型: UNet")
    print(f"{'='*50}")
    
    # 计算模型总参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"不可训练参数量: {total_params - trainable_params:,}")
    
    print(f"{'='*50}\n")

@torch.no_grad()
def validate(model, val_loader, device, criterion):
    """验证函数"""
    model.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0
    num_samples = 0
    
    try:
        progress_bar = tqdm(
            val_loader,
            desc='Validating',
            unit='batch',
            ncols=100,
            bar_format='{l_bar}{bar:30}{r_bar}',
            dynamic_ncols=True,
            leave=True
        )
        
        for batch_idx, data in enumerate(progress_bar):
            try:
                images = data['image'].to(device)
                masks = data['label'].to(device)
                
                # 前向传播
                outputs = model(images)
                
                # 计算损失
                loss = criterion(outputs, masks)
                
                # 计算指标
                iou, dice = calculate_metrics(outputs, masks)
                
                total_loss += loss.item()
                total_iou += iou
                total_dice += dice
                num_samples += 1
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'iou': f'{iou:.4f}',
                    'dice': f'{dice:.4f}'
                })
                
            except Exception as e:
                print(f"处理验证批次 {batch_idx} 时出错: {str(e)}")
                continue
        
        if num_samples == 0:
            raise ValueError("验证过程中没有成功处理任何样本")
            
        return total_loss/num_samples, total_iou/num_samples, total_dice/num_samples
        
    except Exception as e:
        print(f"验证过程中出错: {str(e)}")
        return float('inf'), 0.0, 0.0

def create_model(model_type, num_classes, pretrained=False):
    """创建UNet模型"""
    if model_type == "unet_base":
        model = unet_base(in_channels=3, n_class=num_classes)
    elif model_type == "unet_resnet18":
        model = unet_resnet('resnet18', in_channels=3, n_class=num_classes, pretrained=pretrained)
    elif model_type == "unet_resnet34":
        model = unet_resnet('resnet34', in_channels=3, n_class=num_classes, pretrained=pretrained)
    elif model_type == "unet_resnet50":
        model = unet_resnet('resnet50', in_channels=3, n_class=num_classes, pretrained=pretrained)
    elif model_type == "unet_resnet101":
        model = unet_resnet('resnet101', in_channels=3, n_class=num_classes, pretrained=pretrained)
    elif model_type == "unet_resnet152":
        model = unet_resnet('resnet152', in_channels=3, n_class=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model

def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建训练和验证数据集
    print("正在加载数据集...")
    train_dataset = FullDataset(args.train_image_path, args.train_mask_path, args.input_size, mode='train')
    val_dataset = FullDataset(args.val_image_path, args.val_mask_path, args.input_size, mode='val')
    
    print(f"训练数据集大小: {len(train_dataset)}")
    print(f"验证数据集大小: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        raise ValueError("训练数据集为空！请检查数据路径。")
    if len(val_dataset) == 0:
        raise ValueError("验证数据集为空！请检查数据路径。")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # 测试数据加载
    print("测试数据加载...")
    try:
        test_data = next(iter(train_loader))
        print(f"数据加载测试通过，批次形状: {test_data['image'].shape}, {test_data['label'].shape}")
    except Exception as e:
        print(f"数据加载测试失败: {str(e)}")
        raise
    
    # 创建UNet模型
    print(f"正在创建{args.model_type}模型...")
    try:
        model = create_model(args.model_type, args.num_classes, args.pretrained)
        model = model.to(device)
        print("UNet模型创建成功")
    except Exception as e:
        print(f"UNet模型创建失败: {str(e)}")
        raise
    
    # 如果指定了冻结backbone（仅对ResNet模型有效）
    if args.freeze_backbone and 'resnet' in args.model_type:
        for name, param in model.named_parameters():
            if 'encoder' in name and 'layer' in name:
                param.requires_grad = False
        print("已冻结ResNet backbone参数")
    
    # 打印模型结构
    print_model_structure(model)
    
    # 只优化需要梯度的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("没有可训练的参数！请检查模型结构或冻结设置。")
    
    print(f"\n可训练参数数量: {len(trainable_params)}")
    
    # 损失函数选择
    criterion = bce_dice_loss  # 默认使用BCE+Dice损失
    
    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=1.0e-7)
    scaler = torch.cuda.amp.GradScaler()  # 混合精度
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)

    # 创建CSV文件记录训练指标
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.save_path, f'unet_training_metrics_{timestamp}.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'train_iou', 'train_dice', 'val_loss', 'val_iou', 'val_dice', 'learning_rate'])

    best_val_loss = float('inf')
    best_val_iou = 0
    
    # 训练循环
    for epoch in range(args.epoch):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epoch}")
        print(f"{'='*50}")
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_iou = 0
        train_dice = 0
        num_train_batches = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f'Training Epoch {epoch+1}',
            unit='batch',
            ncols=100,
            bar_format='{l_bar}{bar:30}{r_bar}',
            dynamic_ncols=True,
            leave=True
        )
        
        for batch_idx, data in enumerate(progress_bar):
            try:
                images = data['image'].to(device)
                masks = data['label'].to(device)
                
                with torch.cuda.amp.autocast():
                    # 前向传播
                    outputs = model(images)
                    
                    # 计算损失
                    loss = criterion(outputs, masks)
                    
                    # 计算指标
                    iou, dice = calculate_metrics(outputs, masks)

                # 反向传播
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                train_iou += iou
                train_dice += dice
                num_train_batches += 1

                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'iou': f'{iou:.4f}',
                    'dice': f'{dice:.4f}'
                })
                    
            except Exception as e:
                print(f"处理训练批次 {batch_idx} 时出错: {str(e)}")
                continue

        avg_train_loss = train_loss / num_train_batches if num_train_batches > 0 else 0
        avg_train_iou = train_iou / num_train_batches if num_train_batches > 0 else 0
        avg_train_dice = train_dice / num_train_batches if num_train_batches > 0 else 0
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 验证阶段
        val_loss, val_iou, val_dice = validate(model, val_loader, device, criterion)
        
        # 打印训练和验证指标
        print(f"\n训练指标:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Train IoU: {avg_train_iou:.4f}")
        print(f"Train Dice: {avg_train_dice:.4f}")
        print(f"\n验证指标:")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val IoU: {val_iou:.4f}")
        print(f"Val Dice: {val_dice:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # 记录指标到CSV
        csv_writer.writerow([epoch+1, avg_train_loss, avg_train_iou, avg_train_dice, 
                        val_loss, val_iou, val_dice, current_lr])
        csv_file.flush()  # 确保数据写入文件
        
        # 保存最佳模型
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            model_name = f'unet-best.pth'
            torch.save(model.state_dict(), 
                    os.path.join(args.save_path, model_name))
            print(f'\n保存最佳模型，IoU: {best_val_iou:.4f}')
        
        # 定期保存检查点
        if (epoch+1) % 20 == 0 or (epoch+1) == args.epoch:
            model_name = f'unet-{epoch+1}.pth'
            torch.save(model.state_dict(), 
                    os.path.join(args.save_path, model_name))
            print(f'保存检查点: {model_name}')
    
    csv_file.close()
    print(f'\n训练指标已保存到: {csv_path}')

if __name__ == "__main__":
    main(args)
