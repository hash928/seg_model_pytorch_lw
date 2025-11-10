#!/usr/bin/env python3
# 损失函数模块

import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-6):
    """
    Dice损失函数
    """
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def bce_loss(pred, target, smooth=1e-6):
    """
    BCE_logits损失函数，适合处理二分类问题的损失函数
    BCE损失函数 F.binary_cross_entropy
    """
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
    return loss

def bce_dice_loss(pred, target, bce_weight=0.5, dice_weight=0.5):
    """
    BCE + Dice组合损失函数
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce_weight * bce + dice_weight * dice

def focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Focal Loss for binary/multi-label classification
    pred: logits (not sigmoid)
    target: same shape as pred (0 or 1)
    alpha: balance factor between positive/negative
    gamma: focusing parameter
    reduction: 'none' | 'mean' | 'sum'
    """
    # 1. 计算带logits的 BCE
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    
    # 2. 计算 p_t
    p_t = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)
    
    # 3. 调制项 (1 - p_t)^gamma
    focal_term = (1 - p_t) ** gamma
    
    # 4. 计算 alpha 平衡
    if alpha is not None:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * focal_term * ce_loss
    else:
        loss = focal_term * ce_loss

    # 5. reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

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



