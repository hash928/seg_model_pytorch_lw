#!/usr/bin/env python3
# UNet 模型测试脚本
# 基于SAM2测试代码改编的UNet测试脚本

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime
import random
import re
import sys
from torch.utils.data import DataLoader

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 导入UNet模型
from models.UNetb import unet_base, unet_resnet
from dataset import FullDataset

def parse_shell_args(shell_file='test_unet.sh'):
    """从shell脚本中解析参数，适配 test_unet.sh 的参数格式"""
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
        # 匹配 --xxx "yyy" 或 --xxx
        pattern = r'--(\w+)(?:\s+"([^"]+)")?'
        matches = re.findall(pattern, content)
        for key, value in matches:
            # 处理布尔参数（如 --visualize、--save_predictions）
            if value == '':
                args_dict[key] = True
            else:
                # 转换数值类型
                if value is not None and value.isdigit():
                    value = int(value)
                elif value is not None and re.match(r'^-?\d*\.\d+$', value):
                    value = float(value)
                args_dict[key] = value
        print("从shell脚本读取的参数：")
        for key, value in args_dict.items():
            print(f"{key}: {value}")
    except FileNotFoundError:
        print(f"警告：未找到shell脚本 {shell_file}，将使用命令行参数")
    except Exception as e:
        print(f"警告：解析shell脚本时出错：{e}，将使用命令行参数")
    return args_dict

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser("UNet Model Testing")
    parser.add_argument("--model_path", type=str, required=True,
                        help="训练好的UNet模型路径")
    parser.add_argument("--model_type", type=str, default="unet_resnet18", 
                        choices=["unet_base", "unet_resnet18", "unet_resnet34", "unet_resnet50", "unet_resnet101", "unet_resnet152"],
                        help="UNet模型类型")
    parser.add_argument("--test_image_path", type=str, required=True,
                        help="测试图像路径")
    parser.add_argument("--test_mask_path", type=str, required=True,
                        help="测试标注路径")
    parser.add_argument("--save_path", type=str, default="./test_results",
                        help="测试结果保存路径")
    parser.add_argument("--visualize", action="store_true",
                        help="是否保存可视化结果")
    parser.add_argument("--input_size", type=int, default=352,
                        help="输入图像尺寸")
    parser.add_argument("--num_classes", type=int, default=1,
                        help="分割类别数（1为二值分割）")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="测试批次大小")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="预测阈值")
    
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
    
    return parser.parse_args()

def calculate_metrics(pred, mask, threshold=0.5):
    """计算IoU和Dice指标"""
    pred = (pred > threshold).float()
    intersection = (pred * mask).sum()
    union = (pred + mask).sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred.sum() + mask.sum() + 1e-6)
    return iou.item(), dice.item()

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

def test_single_batch(model, images, masks, device, threshold=0.5):
    """测试单个批次"""
    model.eval()
    with torch.no_grad():
        # 前向传播
        outputs = model(images)
        
        # 应用sigmoid激活
        predictions = torch.sigmoid(outputs)
        
        # 计算指标
        batch_iou = []
        batch_dice = []
        
        for i in range(images.size(0)):
            pred = predictions[i:i+1]
            mask = masks[i:i+1]
            iou, dice = calculate_metrics(pred, mask, threshold)
            batch_iou.append(iou)
            batch_dice.append(dice)
        
        return predictions.cpu().numpy(), batch_iou, batch_dice

def visualize_result(image, gt_mask, pred_mask, iou, dice, save_path, sample_idx, original_mask_path=None, mask_name=None):
    """可视化测试结果"""
    plt.figure(figsize=(15, 5))
    
    # 确保图像数据格式正确
    if len(image.shape) == 3 and image.shape[0] == 1:
        image = image.squeeze(0)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = image.squeeze(-1)
    
    # 确保掩码数据格式正确
    if len(gt_mask.shape) == 3 and gt_mask.shape[0] == 1:
        gt_mask = gt_mask.squeeze(0)
    elif len(gt_mask.shape) == 3 and gt_mask.shape[2] == 1:
        gt_mask = gt_mask.squeeze(-1)
    
    if len(pred_mask.shape) == 3 and pred_mask.shape[0] == 1:
        pred_mask = pred_mask.squeeze(0)
    elif len(pred_mask.shape) == 3 and pred_mask.shape[2] == 1:
        pred_mask = pred_mask.squeeze(-1)
    
    # 原始图像
    plt.subplot(1, 4, 1)
    if len(image.shape) == 3:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # 真实标注
    plt.subplot(1, 4, 2)
    plt.imshow(gt_mask, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # 预测结果
    plt.subplot(1, 4, 3)
    plt.imshow(pred_mask, cmap='gray')
    plt.title(f'Prediction\nIoU: {iou:.3f}, Dice: {dice:.3f}')
    plt.axis('off')
    
    # 叠加显示
    plt.subplot(1, 4, 4)
    if len(image.shape) == 3:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')
    # 叠加预测结果（红色）
    pred_overlay = np.ma.masked_where(pred_mask < 0.5, pred_mask)
    plt.imshow(pred_overlay, cmap='Reds', alpha=0.6)
    # 叠加真实标注（绿色）
    gt_overlay = np.ma.masked_where(gt_mask < 0.5, gt_mask)
    plt.imshow(gt_overlay, cmap='Greens', alpha=0.6)
    plt.title('Overlay\nRed: Pred, Green: GT')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'sample_{sample_idx:04d}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 单独保存预测结果图像，调整到原始标签尺寸
    pred_img = (pred_mask * 255).astype(np.uint8)
    
    # 如果提供了原始标签路径，读取原始尺寸并调整预测结果
    if original_mask_path and os.path.exists(original_mask_path):
        try:
            original_mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)
            if original_mask is not None:
                # 调整预测结果到原始标签尺寸
                pred_img = cv2.resize(pred_img, (original_mask.shape[1], original_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            print(f"警告: 无法读取原始标签 {original_mask_path}: {e}")
    
    # 保存到predictions目录，使用与原始标签相同的文件名
    base_dir = os.path.dirname(save_path)
    predictions_dir = os.path.join(base_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    if mask_name:
        pred_filename = mask_name
    else:
        pred_filename = f"prediction_{sample_idx:04d}.png"
    
    cv2.imwrite(os.path.join(predictions_dir, pred_filename), pred_img)

def batch_test(model, test_loader, device, args):
    """批量测试"""
    print("开始批量测试...")
    
    # 创建保存目录
    if args.visualize:
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(os.path.join(args.save_path, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(args.save_path, 'predictions'), exist_ok=True)
    
    # 测试所有样本
    num_batches = len(test_loader)
    
    results = []
    total_iou = 0
    total_dice = 0
    valid_samples = 0
    
    progress_bar = tqdm(test_loader, desc='Testing', unit='batch', ncols=150)
    
    for batch_idx, data in enumerate(progress_bar):
        try:
            images = data['image'].to(device)
            masks = data['label'].to(device)
            
            # 测试批次
            predictions, batch_iou, batch_dice = test_single_batch(model, images, masks, device, args.threshold)
            
            # 处理每个样本的结果
            for i in range(images.size(0)):
                # 获取原始标签路径和文件名
                sample_idx = batch_idx * args.batch_size + i
                if sample_idx < len(test_loader.dataset):
                    original_mask_path = test_loader.dataset.gts[sample_idx]
                    mask_name = os.path.basename(original_mask_path)
                else:
                    original_mask_path = None
                    mask_name = None
                
                # 反归一化图像用于可视化
                image_np = images[i].detach().cpu().numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = image_np * std + mean
                image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
                
                # 获取掩码
                gt_mask = masks[i].detach().cpu().numpy()
                pred_mask = predictions[i]
                
                # 处理掩码形状
                if len(gt_mask.shape) == 3 and gt_mask.shape[0] == 1:
                    gt_mask = gt_mask.squeeze(0)
                if len(pred_mask.shape) == 3 and pred_mask.shape[0] == 1:
                    pred_mask = pred_mask.squeeze(0)
                
                result = {
                    'sample_idx': sample_idx,
                    'iou': batch_iou[i],
                    'dice': batch_dice[i],
                    'prediction': pred_mask,
                    'gt_mask': gt_mask,
                    'image': image_np,
                    'mask_name': mask_name,
                    'original_mask_path': original_mask_path
                }
                
                results.append(result)
                total_iou += batch_iou[i]
                total_dice += batch_dice[i]
                valid_samples += 1
                
                # 可视化结果
                if args.visualize:
                    save_path = os.path.join(args.save_path, 'visualizations')
                    visualize_result(image_np, gt_mask, pred_mask, batch_iou[i], batch_dice[i], 
                                   save_path, sample_idx, original_mask_path, mask_name)
                
                # 更新进度条
                progress_bar.set_postfix({
                    'IoU': f'{batch_iou[i]:.3f}',
                    'Dice': f'{batch_dice[i]:.3f}',
                    'Avg_IoU': f'{total_iou/valid_samples:.3f}',
                    'Avg_Dice': f'{total_dice/valid_samples:.3f}'
                })
                    
        except Exception as e:
            print(f"处理批次 {batch_idx} 时出错: {str(e)}")
            continue
    
    # 计算平均指标
    if valid_samples > 0:
        avg_iou = total_iou / valid_samples
        avg_dice = total_dice / valid_samples
    else:
        avg_iou = 0
        avg_dice = 0
    
    return results, avg_iou, avg_dice, valid_samples

def save_test_results(results, avg_iou, avg_dice, valid_samples, args):
    """保存测试结果到CSV文件"""
    # 保存为CSV文件
    csv_file = os.path.join(args.save_path, 'test_results.csv')
    with open(csv_file, 'w') as f:
        f.write('sample_idx,iou,dice,prediction_image\n')
        for result in results:
            # 获取对应的原始标签文件名
            if result['mask_name']:
                pred_image_path = f'predictions/{result["mask_name"]}'
            else:
                pred_image_path = f'predictions/prediction_{result["sample_idx"]:04d}.png'
            f.write(f'{result["sample_idx"]},{result["iou"]:.6f},{result["dice"]:.6f},{pred_image_path}\n')
    
    print(f"\n测试结果已保存到: {args.save_path}")
    print(f"CSV结果: {csv_file}")
    
    # 保存预测结果统计信息
    if args.visualize:
        stats = {
            'total_samples': len(results),
            'valid_samples': valid_samples,
            'average_iou': avg_iou,
            'average_dice': avg_dice,
            'iou_scores': [result['iou'] for result in results],
            'dice_scores': [result['dice'] for result in results],
            'prediction_images_dir': 'predictions',
            'visualizations_dir': 'visualizations'
        }
        
        stats_file = os.path.join(args.save_path, 'test_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"统计信息: {stats_file}")

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

def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载测试数据集
    print("正在加载测试数据集...")
    test_dataset = FullDataset(args.test_image_path, args.test_mask_path, args.input_size, mode='val')
    print(f"测试数据集大小: {len(test_dataset)}")
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    # 创建模型
    print(f"正在创建{args.model_type}模型...")
    try:
        model = create_model(args.model_type, args.num_classes, pretrained=False)
        model = model.to(device)
        print("UNet模型创建成功")
    except Exception as e:
        print(f"UNet模型创建失败: {str(e)}")
        raise
    
    # 打印模型结构
    print_model_structure(model)
    
    # 加载训练好的权重
    print(f"正在加载训练好的权重: {args.model_path}")
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("模型权重加载成功")
    except Exception as e:
        print(f"模型权重加载失败: {str(e)}")
        raise
    
    # 设置为评估模式
    model.eval()
    
    # 批量测试
    results, avg_iou, avg_dice, valid_samples = batch_test(model, test_loader, device, args)
    
    # 打印测试结果
    print(f"\n{'='*50}")
    print("测试结果总结")
    print(f"{'='*50}")
    print(f"总样本数: {len(results)}")
    print(f"有效样本数: {valid_samples}")
    print(f"平均IoU: {avg_iou:.4f}")
    print(f"平均Dice: {avg_dice:.4f}")
    print(f"使用阈值: {args.threshold}")
    print(f"{'='*50}")
    
    # 保存测试结果
    save_test_results(results, avg_iou, avg_dice, valid_samples, args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
