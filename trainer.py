#!/usr/bin/env python3
# 训练器模块

import os
import torch
import torch.optim as opt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
from datetime import datetime

from utils import bce_dice_loss, calculate_metrics, focal_loss, dice_loss, bce_loss
from model_utils import create_model, print_model_structure, freeze_backbone
from utils.tensorboard_logger import TensorBoardLogger

class UNetTrainer:
    """UNet训练器类"""
    
    def __init__(self, args, train_dataset, val_dataset, generator=None, worker_init_fn=None):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            generator=generator,  # 使用固定的generator确保shuffle可复现
            worker_init_fn=worker_init_fn  # 为每个worker设置随机种子
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=worker_init_fn  # 验证时也使用相同的worker初始化
        )
        
        # 创建模型
        self.model = self._create_model()
        
        # 设置优化器和调度器
        self.optimizer, self.scheduler, self.scaler = self._setup_optimizer()
        
        # 初始化 TensorBoard 记录器
        try:
            self.tb_logger = TensorBoardLogger(args.save_path, self.model, self.device, args.input_size)
        except Exception as e:
            print(f"初始化 TensorBoardLogger 失败: {str(e)}")
            self.tb_logger = None
        
        # 创建保存目录
        os.makedirs(args.save_path, exist_ok=True)
        
        # 初始化训练记录
        self._init_training_log()
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.best_val_iou = 0
        
    def _create_model(self):
        """创建模型"""
        print(f"正在创建{self.args.model_type}模型...")
        try:
            # 对于非UNet-ResNet模型，pretrained参数可能不适用
            if 'unet' in self.args.model_type and 'resnet' in self.args.model_type:
                model = create_model(
                    self.args.model_type, 
                    self.args.num_classes, 
                    self.args.pretrained, 
                    self.args.use_aspp, 
                    getattr(self.args, 'use_se', False),
                    getattr(self.args, 'use_cbam', False),
                    getattr(self.args, 'use_ca', False),
                    getattr(self.args, 'use_eca', False),
                    getattr(self.args, 'attention_type', None)
                )
            elif 'unet' in self.args.model_type:
                # UNet base模型
                model = create_model(
                    self.args.model_type, 
                    self.args.num_classes, 
                    False, 
                    self.args.use_aspp, 
                    getattr(self.args, 'use_se', False),
                    getattr(self.args, 'use_cbam', False),
                    getattr(self.args, 'use_ca', False),
                    getattr(self.args, 'use_eca', False),
                    getattr(self.args, 'attention_type', None)
                )
            else:
                # FCN8s和DeepLabV3+模型不使用pretrained、ASPP和SE参数
                model = create_model(
                    self.args.model_type,
                    self.args.num_classes,
                    False,
                    False,
                    False,
                    xception_width_mult=getattr(self.args, "xception_width_mult", 1.0),
                    xception_output_stride=getattr(self.args, "xception_output_stride", 16),
                )
            
            model = model.to(self.device)
            print(f"{self.args.model_type}模型创建成功")
            
            # 如果指定了冻结backbone
            if self.args.freeze_backbone:
                freeze_backbone(model, self.args.model_type)
            
            # 打印模型结构
            print_model_structure(model, self.args.model_type)
            
            return model
        except Exception as e:
            print(f"{self.args.model_type}模型创建失败: {str(e)}")
            raise
    
    def _setup_optimizer(self):
        """设置优化器和学习率调度器"""
        # 只优化需要梯度的参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError("没有可训练的参数！请检查模型结构或冻结设置。")
        
        print(f"\n可训练参数数量: {len(trainable_params)}")
        
        # 优化器和学习率调度器
        optimizer = torch.optim.AdamW(trainable_params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, self.args.epoch, eta_min=1.0e-7)
        device_type = str(self.device).split(':')[0]
        scaler = torch.amp.GradScaler(device_type)  # 混合精度
        
        return optimizer, scheduler, scaler
    
    def _init_training_log(self):
        """初始化训练记录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.args.save_path, f'unet_training_metrics_{timestamp}.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['epoch', 'train_loss', 'train_iou', 'train_dice', 'val_loss', 'val_iou', 'val_dice', 'learning_rate'])
    
    def validate(self):
        """验证函数"""
        self.model.eval()
        total_loss = 0
        total_iou = 0
        total_dice = 0
        num_samples = 0
        
        try:
            progress_bar = tqdm(
                self.val_loader,
                desc='Validating',
                unit='batch',
                ncols=100,
                bar_format='{l_bar}{bar:30}{r_bar}',
                dynamic_ncols=True,
                leave=True
            )
            
            for batch_idx, data in enumerate(progress_bar):
                try:
                    images = data['image'].to(self.device)
                    masks = data['label'].to(self.device)
                    
                    # 前向传播
                    outputs = self.model(images)
                    
                    # 计算损失
                    loss = dice_loss(outputs, masks)
                    
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
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        train_loss = 0
        train_iou = 0
        train_dice = 0
        num_train_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f'Training Epoch {epoch+1}',
            unit='batch',
            ncols=100,
            bar_format='{l_bar}{bar:30}{r_bar}',
            dynamic_ncols=True,
            leave=True
        )
        
        for batch_idx, data in enumerate(progress_bar):
            try:
                images = data['image'].to(self.device)
                masks = data['label'].to(self.device)

                device_type = str(self.device).split(':')[0]
                with torch.amp.autocast(device_type):
                    # 前向传播
                    outputs = self.model(images)
                    
                    # 计算损失
                    loss = dice_loss(outputs, masks)
                    
                    # 计算指标
                    iou, dice = calculate_metrics(outputs, masks)

                # 反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                train_loss += loss.item()
                train_iou += iou
                train_dice += dice
                num_train_batches += 1
                
                # 记录训练过程中的指标到 TensorBoard（按batch）
                if self.tb_logger is not None:
                    try:
                        self.tb_logger.log_training_metrics(loss.item(), iou, dice)
                    except Exception as e:
                        print(f"记录训练指标到 TensorBoard 失败: {str(e)}")

                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'iou': f'{iou:.4f}',
                    'dice': f'{dice:.4f}'
                })
                    
            except Exception as e:
                print(f"处理训练批次 {batch_idx} 时出错: {str(e)}")
                continue

        return train_loss / num_train_batches if num_train_batches > 0 else 0, \
               train_iou / num_train_batches if num_train_batches > 0 else 0, \
               train_dice / num_train_batches if num_train_batches > 0 else 0
    
    def save_model(self, epoch, val_iou):
        """保存模型"""
        # 保存最佳模型
        if val_iou > self.best_val_iou:
            self.best_val_iou = val_iou
            model_name = f'{self.args.model_type}-best.pth'
            torch.save(self.model.state_dict(), 
                    os.path.join(self.args.save_path, model_name))
            print(f'\n保存最佳模型，IoU: {self.best_val_iou:.4f}')
        
        # 定期保存检查点
        if (epoch+1) % 20 == 0 or (epoch+1) == self.args.epoch:
            model_name = f'{self.args.model_type}-{epoch+1}.pth'
            torch.save(self.model.state_dict(), 
                    os.path.join(self.args.save_path, model_name))
            print(f'保存检查点: {model_name}')
    
    def train(self):
        """主训练循环"""
        print(f"开始训练，共{self.args.epoch}个epoch")
        
        for epoch in range(self.args.epoch):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.args.epoch}")
            print(f"{'='*50}")
            
            # 训练阶段
            avg_train_loss, avg_train_iou, avg_train_dice = self.train_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # 验证阶段
            val_loss, val_iou, val_dice = self.validate()
            
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
            self.csv_writer.writerow([epoch+1, avg_train_loss, avg_train_iou, avg_train_dice, 
                            val_loss, val_iou, val_dice, current_lr])
            self.csv_file.flush()  # 确保数据写入文件
            
            # 记录 epoch 级别的摘要到 TensorBoard
            if self.tb_logger is not None:
                try:
                    self.tb_logger.log_epoch_summary(
                        epoch,
                        avg_train_loss, avg_train_iou, avg_train_dice,
                        val_loss, val_iou, val_dice,
                        current_lr
                    )
                    # 可选：定期记录样本预测
                    if (epoch + 1) % 5 == 0:
                        self.tb_logger.log_sample_images(epoch, self.val_loader)
                except Exception as e:
                    print(f"记录 TensorBoard Epoch 摘要失败: {str(e)}")
            
            # 保存模型
            self.save_model(epoch, val_iou)
        
        # 关闭CSV文件
        self.csv_file.close()
        
        # 关闭 TensorBoard
        if self.tb_logger is not None:
            try:
                self.tb_logger.close()
            except Exception as e:
                print(f"关闭 TensorBoardLogger 失败: {str(e)}")
        print(f'\n训练指标已保存到: {self.csv_path}')
        print("训练完成！")
