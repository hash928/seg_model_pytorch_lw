#!/usr/bin/env python3
# TensorBoard 记录器模块

import os
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import io
from PIL import Image


class TensorBoardLogger:
    """TensorBoard 记录器类"""
    
    def __init__(self, save_path, model, device, input_size):
        """
        初始化 TensorBoard 记录器
        
        Args:
            save_path: 保存路径
            model: 模型
            device: 设备
            input_size: 输入图像尺寸
        """
        self.save_path = save_path
        self.model = model
        self.device = device
        self.input_size = input_size
        self.global_step = 0
        
        # 创建 TensorBoard 日志目录
        self.log_dir = os.path.join(save_path, "runs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # 记录模型图
        self._log_model_graph()
    
    def _log_model_graph(self):
        """记录模型图到 TensorBoard"""
        try:
            # 创建虚拟输入来追踪模型图
            dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
            self.writer.add_graph(self.model, dummy_input)
            print("模型图已记录到 TensorBoard")
        except Exception as e:
            print(f"记录模型图时出错: {str(e)}")
            print("这不会影响训练，但 TensorBoard Graphs 功能可能不可用")
    
    def log_training_metrics(self, loss, iou, dice):
        """记录训练指标"""
        self.global_step += 1
        self.writer.add_scalar("Train/Loss", loss, self.global_step)
        self.writer.add_scalar("Train/IoU", iou, self.global_step)
        self.writer.add_scalar("Train/Dice", dice, self.global_step)
    
    def log_validation_metrics(self, loss, iou, dice, epoch):
        """记录验证指标"""
        self.writer.add_scalar("Val/Loss", loss, epoch)
        self.writer.add_scalar("Val/IoU", iou, epoch)
        self.writer.add_scalar("Val/Dice", dice, epoch)
    
    def log_learning_rate(self, lr, epoch):
        """记录学习率"""
        self.writer.add_scalar("Learning_Rate", lr, epoch)
    
    def log_model_parameters(self, epoch):
        """记录模型参数分布"""
        try:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # 记录参数分布直方图
                    self.writer.add_histogram(f"Parameters/{name}", param, epoch)
                    # 记录梯度分布（如果存在）
                    if param.grad is not None:
                        self.writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
        except Exception as e:
            print(f"记录模型参数时出错: {str(e)}")
    
    def log_sample_images(self, epoch, val_loader):
        """记录样本图像和预测结果"""
        try:
            self.model.eval()
            with torch.no_grad():
                # 获取一个验证批次
                data = next(iter(val_loader))
                images = data['image'][:4].to(self.device)  # 只取前4个样本
                masks = data['label'][:4].to(self.device)
                
                # 获取预测结果
                outputs = self.model(images)
                predictions = torch.sigmoid(outputs)
                
                # 反归一化图像用于显示
                images_denorm = self._denormalize_images(images)
                
                # 创建图像网格
                grid_images = []
                for i in range(4):
                    # 原始图像
                    img = images_denorm[i].cpu()
                    # 真实掩码
                    gt_mask = masks[i].cpu().repeat(3, 1, 1)
                    # 预测掩码
                    pred_mask = predictions[i].cpu().repeat(3, 1, 1)
                    
                    # 组合图像
                    combined = torch.cat([img, gt_mask, pred_mask], dim=2)
                    grid_images.append(combined)
                
                # 创建网格
                grid = torchvision.utils.make_grid(grid_images, nrow=1, padding=2)
                self.writer.add_image(f"Sample_Predictions/Epoch_{epoch+1}", grid, epoch)
                
        except Exception as e:
            print(f"记录样本图像时出错: {str(e)}")
        finally:
            self.model.train()
    
    def _denormalize_images(self, images):
        """反归一化图像"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        return torch.clamp(images * std + mean, 0, 1)
    
    def log_epoch_summary(self, epoch, train_loss, train_iou, train_dice, 
                          val_loss, val_iou, val_dice, lr):
        """记录一个 epoch 的完整摘要"""
        # 记录验证指标
        self.log_validation_metrics(val_loss, val_iou, val_dice, epoch)
        self.log_learning_rate(lr, epoch)
        
        # 每10个epoch记录一次详细内容
        if (epoch + 1) % 10 == 0:
            self.log_model_parameters(epoch)
    
    def close(self):
        """关闭 TensorBoard writer"""
        self.writer.close()
        print(f"TensorBoard 日志已保存到: {self.log_dir}")