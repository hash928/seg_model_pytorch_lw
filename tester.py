#!/usr/bin/env python3
# 测试器模块

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_utils import create_model, print_model_structure
from utils import test_calculate_metrics, visualize_result, save_test_results, denormalize_image
from dataset import FullDataset

class UNetTester:
    """UNet测试器类"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建测试数据集
        self.test_dataset = FullDataset(args.test_image_path, args.test_mask_path, args.input_size, mode='val')
        print(f"测试数据集大小: {len(self.test_dataset)}")
        
        # 创建数据加载器
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        
        # 创建模型
        self.model = self._create_model()
        
        # 加载模型权重
        self._load_model_weights()
        
        # 创建保存目录
        if args.visualize:
            os.makedirs(args.save_path, exist_ok=True)
            os.makedirs(os.path.join(args.save_path, 'visualizations'), exist_ok=True)
            os.makedirs(os.path.join(args.save_path, 'predictions'), exist_ok=True)
    
    def _create_model(self):
        """创建模型"""
        print(f"正在创建{self.args.model_type}模型...")
        try:
            # 对于非UNet-ResNet模型，pretrained参数可能不适用
            if 'unet' in self.args.model_type and 'resnet' in self.args.model_type:
                model = create_model(self.args.model_type, self.args.num_classes, pretrained=False, use_aspp=getattr(self.args, 'use_aspp', False), use_se=getattr(self.args, 'use_se', False))
            elif 'unet' in self.args.model_type:
                # UNet base模型
                model = create_model(self.args.model_type, self.args.num_classes, False, use_aspp=getattr(self.args, 'use_aspp', False), use_se=getattr(self.args, 'use_se', False))
            else:
                # FCN8s和DeepLabV3+模型不使用pretrained、ASPP和SE参数
                model = create_model(self.args.model_type, self.args.num_classes, False, False, False)
            
            model = model.to(self.device)
            print(f"{self.args.model_type}模型创建成功")
            
            # 打印模型结构
            print_model_structure(model, self.args.model_type)
            
            return model
        except Exception as e:
            print(f"{self.args.model_type}模型创建失败: {str(e)}")
            raise
    
    def _load_model_weights(self):
        """加载模型权重"""
        print(f"正在加载训练好的权重: {self.args.model_path}")
        try:
            state_dict = torch.load(self.args.model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            print("模型权重加载成功")
        except Exception as e:
            print(f"模型权重加载失败: {str(e)}")
            raise
        
        # 设置为评估模式
        self.model.eval()
    
    def test_single_batch(self, images, masks):
        """测试单个批次"""
        with torch.no_grad():
            # 前向传播
            outputs = self.model(images)
            
            # 应用sigmoid激活
            predictions = torch.sigmoid(outputs)
            
            # 计算指标
            batch_iou = []
            batch_dice = []
            
            for i in range(images.size(0)):
                pred = predictions[i:i+1]
                mask = masks[i:i+1]
                iou, dice = test_calculate_metrics(pred, mask, self.args.threshold)
                batch_iou.append(iou)
                batch_dice.append(dice)
            
            return predictions.cpu().numpy(), batch_iou, batch_dice
    
    def test(self):
        """执行测试"""
        print("开始批量测试...")
        
        results = []
        total_iou = 0
        total_dice = 0
        valid_samples = 0
        
        progress_bar = tqdm(self.test_loader, desc='Testing', unit='batch', ncols=120)
        
        for batch_idx, data in enumerate(progress_bar):
            try:
                images = data['image'].to(self.device)
                masks = data['label'].to(self.device)
                
                # 测试批次
                predictions, batch_iou, batch_dice = self.test_single_batch(images, masks)
                
                # 处理每个样本的结果
                for i in range(images.size(0)):
                    # 获取原始标签路径和文件名
                    sample_idx = batch_idx * self.args.batch_size + i
                    if sample_idx < len(self.test_loader.dataset):
                        original_mask_path = self.test_loader.dataset.gts[sample_idx]
                        mask_name = os.path.basename(original_mask_path)
                    else:
                        original_mask_path = None
                        mask_name = None
                    
                    # 反归一化图像用于可视化
                    image_np = denormalize_image(images[i])
                    
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
                    if self.args.visualize:
                        save_path = os.path.join(self.args.save_path, 'visualizations')
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
    
    def print_results(self, results, avg_iou, avg_dice, valid_samples):
        """打印测试结果"""
        print(f"\n{'='*50}")
        print("测试结果总结")
        print(f"{'='*50}")
        print(f"总样本数: {len(results)}")
        print(f"有效样本数: {valid_samples}")
        print(f"平均IoU: {avg_iou:.4f}")
        print(f"平均Dice: {avg_dice:.4f}")
        print(f"使用阈值: {self.args.threshold}")
        print(f"{'='*50}")
    
    def save_results(self, results, avg_iou, avg_dice, valid_samples):
        """保存测试结果"""
        save_test_results(results, avg_iou, avg_dice, valid_samples, 
                         self.args.save_path, self.args.visualize)
