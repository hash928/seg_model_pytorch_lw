#!/usr/bin/env python3
# 测试工具模块

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from datetime import datetime

def calculate_metrics(pred, mask, threshold=0.5):
    """计算IoU和Dice指标"""
    pred = (pred > threshold).float()
    intersection = (pred * mask).sum()
    union = (pred + mask).sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred.sum() + mask.sum() + 1e-6)
    return iou.item(), dice.item()

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

def save_test_results(results, avg_iou, avg_dice, valid_samples, save_path, visualize=False):
    """保存测试结果到CSV文件"""
    # 保存为CSV文件
    csv_file = os.path.join(save_path, 'test_results.csv')
    with open(csv_file, 'w') as f:
        f.write('sample_idx,iou,dice,prediction_image\n')
        for result in results:
            # 获取对应的原始标签文件名
            if result['mask_name']:
                pred_image_path = f'predictions/{result["mask_name"]}'
            else:
                pred_image_path = f'predictions/prediction_{result["sample_idx"]:04d}.png'
            f.write(f'{result["sample_idx"]},{result["iou"]:.6f},{result["dice"]:.6f},{pred_image_path}\n')
    
    print(f"\n测试结果已保存到: {save_path}")
    print(f"CSV结果: {csv_file}")
    
    # 保存预测结果统计信息
    if visualize:
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
        
        stats_file = os.path.join(save_path, 'test_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"统计信息: {stats_file}")

def denormalize_image(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """反归一化图像用于可视化"""
    image_np = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(mean)
    std = np.array(std)
    image_np = image_np * std + mean
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    return image_np
