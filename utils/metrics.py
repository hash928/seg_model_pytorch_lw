#!/usr/bin/env python3
# 指标计算模块

import torch

def calculate_metrics(pred, mask, threshold=0.5):
    """计算IoU和Dice指标"""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * mask).sum()
    union = (pred + mask).sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred.sum() + mask.sum() + 1e-6)
    return iou.item(), dice.item()

def calculate_batch_metrics(pred, mask, threshold=0.5):
    """计算批次级别的IoU和Dice指标"""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3)) + 1e-6)
    return iou.mean().item(), dice.mean().item()
