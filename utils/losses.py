#!/usr/bin/env python3
# 损失函数模块

import torch
import torch.nn.functional as F

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



