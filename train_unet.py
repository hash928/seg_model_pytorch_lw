#!/usr/bin/env python3
# Train UNet on LabPics 1 dataset
# 基于SAM2训练代码改编的UNet训练脚本 - 模块化版本

import os
import torch
from torch.utils.data import DataLoader

# 导入模块化组件
from config import parse_args
from trainer import UNetTrainer
from dataset import FullDataset

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
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
    
    # 测试数据加载
    print("测试数据加载...")
    try:
        test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_data = next(iter(test_loader))
        print(f"数据加载测试通过，批次形状: {test_data['image'].shape}, {test_data['label'].shape}")
    except Exception as e:
        print(f"数据加载测试失败: {str(e)}")
        raise
    
    # 创建训练器并开始训练
    trainer = UNetTrainer(args, train_dataset, val_dataset)
    trainer.train()

if __name__ == "__main__":
    main()
