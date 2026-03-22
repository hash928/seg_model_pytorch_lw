#!/usr/bin/env python3

import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader

# 导入模块化组件
from config import parse_args
from trainer import UNetTrainer
from dataset import FullDataset

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("✅ 检测到 GPU，可使用 CUDA 进行加速。")
else:
    device = torch.device("cpu")
    print("⚠️ 未检测到 GPU，使用 CPU 进行训练。")


def set_random_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn 的 deterministic/benchmark 会在 main() 里根据 args 控制

def get_worker_init_fn(seed):
    """返回worker初始化函数，为每个worker进程设置随机种子"""
    def worker_init_fn(worker_id):
        # 每个worker使用不同的种子，但基于主种子
        # 这样既保证了可复现性，又让不同worker有不同的随机序列
        worker_seed = (seed + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return worker_init_fn

def main():
    """主函数"""

    # 解析参数（需要在设置随机种子之前，因为可能从shell脚本读取参数）
    args = parse_args()
    
    # 设置随机种子（在创建数据集之前设置）
    seed = getattr(args, "seed", 33)
    set_random_seed(seed)

    # 复现 vs 性能：deterministic 更可复现但通常更慢
    if getattr(args, "deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("已启用 cudnn deterministic：训练更可复现，但通常更慢。")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        print("已启用 cudnn benchmark：训练更快，但不同运行间可能略有差异。")
    
    # 创建用于DataLoader的generator，确保shuffle的可复现性
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # 创建worker初始化函数
    worker_init_fn = get_worker_init_fn(seed)
    
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
    trainer = UNetTrainer(args, train_dataset, val_dataset, generator, worker_init_fn)
    trainer.train()

if __name__ == "__main__":
    main()
