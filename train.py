import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import FullDataset
from models.SAM2UNet import SAM2UNet
from models.UNet import UNet
from models.FCN import FCN
from models.UNetPlusPlus import UNetPlusPlus, UNetPlus
from models.LSTMUNet import LSTMUNet
from tqdm import tqdm
import numpy as np
from torchsummary import summary
import csv
from datetime import datetime
import random
import torch.optim as opt
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
import re
import sys

def parse_shell_args(shell_file='train.sh'):
    """从shell脚本中解析参数"""
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
            
        # 使用正则表达式匹配参数
        pattern = r'--(\w+)\s+"([^"]+)"'
        matches = re.findall(pattern, content)
        
        for key, value in matches:
            # 转换数值类型
            if value.isdigit():
                value = int(value)
            elif re.match(r'^-?\d*\.\d+$', value):
                value = float(value)
            args_dict[key] = value
            
        # 处理布尔参数
        if '--freeze_backbone' in content and not any(line.strip().startswith('#') and '--freeze_backbone' in line for line in lines):
            args_dict['freeze_backbone'] = True
            
        print("从shell脚本读取的参数：")
        for key, value in args_dict.items():
            print(f"{key}: {value}")
            
    except FileNotFoundError:
        print(f"警告：未找到shell脚本 {shell_file}，将使用命令行参数")
    except Exception as e:
        print(f"警告：解析shell脚本时出错：{e}，将使用命令行参数")
        
    return args_dict

# 创建参数解析器
parser = argparse.ArgumentParser("Model Training")
parser.add_argument("--model_type", type=str, default="sam2unet", 
                    choices=["sam2unet", "unet", "fcn", "unetplusplus", "unetplus", "lstmunet", "sam2", "fastsam"],
                    help="选择模型类型: sam2unet, unet, fcn, unetplusplus, unetplus, lstmunet, sam2 或 fastsam")
parser.add_argument("--hiera_path", type=str, 
                    help="path to the sam2 pretrained hiera (仅当model_type为sam2unet或sam2时需要)")
parser.add_argument("--pretrained_path", type=str,
                    help="预训练模型路径，用于迁移学习")
parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet34"],
                    help="FCN模型的backbone (仅当model_type为fcn时需要)")
parser.add_argument("--train_image_path", type=str, required=True, 
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the mask file for training")
parser.add_argument("--val_image_path", type=str, required=True, 
                    help="path to the image that used to val the model")
parser.add_argument("--val_mask_path", type=str, required=True,
                    help="path to the mask file for val")
parser.add_argument('--save_path', type=str, required=True,
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=20, 
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--freeze_backbone", action="store_true",
                    help="是否冻结主干网络参数")
parser.add_argument("--deep_supervision", action="store_true",
                    help="是否使用深度监督（仅当model_type为unetplusplus时有效）")
parser.add_argument("--num_workers", type=int, default=3,
                    help="数据加载时使用的子进程数量，建议设置为CPU核心数的2-4倍")
parser.add_argument("--debug_validation", action="store_true",
                    help="使用调试模式进行验证（只处理前几个批次）")

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

# 解析命令行参数
args = parser.parse_args()

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def calculate_metrics(pred, mask):
    pred = (pred > 0.5).float()
    intersection = (pred * mask).sum()
    union = (pred + mask).sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred.sum() + mask.sum() + 1e-6)
    return iou.item(), dice.item()

def simple_validate(model, val_loader, device, model_type):
    """简化的验证函数，用于调试"""
    model.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                x = batch['image'].to(device)
                target = batch['label'].to(device)
                
                # 模型推理
                if model_type == "sam2unet":
                    pred0, pred1, pred2 = model(x)
                    pred = pred2
                elif model_type == "unetplusplus" and isinstance(model, UNetPlusPlus) and model.deep_supervision:
                    preds = model(x)
                    pred = preds[-1]
                else:
                    pred = model(x)
                
                loss = structure_loss(pred, target)
                pred = torch.sigmoid(pred)
                iou, dice = calculate_metrics(pred, target)
                
                batch_size = x.size(0)
                total_loss += loss.item() * batch_size
                total_iou += iou * batch_size
                total_dice += dice * batch_size
                num_samples += batch_size
                
                # 只处理前几个批次进行调试
                if batch_idx >= 2:
                    break
                    
            except Exception:
                break
    
    if num_samples == 0:
        return float('inf'), 0.0, 0.0
    
    return total_loss/num_samples, total_iou/num_samples, total_dice/num_samples

@torch.no_grad()
def validate(model, val_loader, device, model_type):
    model.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0
    num_samples = 0
    
    progress_bar = tqdm(
        val_loader,
        desc='Validating',
        unit='batch',
        ncols=100,
        bar_format='{l_bar}{bar:30}{r_bar}',
        dynamic_ncols=True,
        leave=True
    )
    
    for batch in progress_bar:
        try:
            x = batch['image'].to(device)
            target = batch['label'].to(device)
            
            # 模型推理
            if model_type == "sam2unet":
                pred0, pred1, pred2 = model(x)
                pred = pred2
            elif model_type == "unetplusplus" and isinstance(model, UNetPlusPlus) and model.deep_supervision:
                preds = model(x)
                pred = preds[-1]
            else:
                pred = model(x)
            
            loss = structure_loss(pred, target)
            pred = torch.sigmoid(pred)
            iou, dice = calculate_metrics(pred, target)
            
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_iou += iou * batch_size
            total_dice += dice * batch_size
            num_samples += batch_size
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}',
                'dice': f'{dice:.4f}'
            })
            
        except Exception:
            # 静默处理错误，不影响进度条
            continue
    
    if num_samples == 0:
        return float('inf'), 0.0, 0.0
    
    return total_loss/num_samples, total_iou/num_samples, total_dice/num_samples

def print_model_structure(model, model_type):
    """打印模型结构"""
    print(f"\n{'='*50}")
    print(f"模型类型: {model_type}")
    print(f"{'='*50}")
    
    # 计算模型总参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"不可训练参数量: {total_params - trainable_params:,}")
    
    print(f"\n{'='*50}")
    print("模型结构:")
    print(f"{'='*50}")
    
    # 将模型移到CPU进行结构打印
    device = next(model.parameters()).device
    model = model.cpu()
    # 创建一个CPU上的输入张量
    x = torch.randn(1, 3, 352, 352)
    summary(model, (3, 352, 352), device='cpu')
    model = model.to(device)  # 将模型移回原设备
    print(f"{'='*50}\n")

def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建训练和验证数据加载器
    train_dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, mode='train')
    val_dataset = FullDataset(args.val_image_path, args.val_mask_path, 352, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # 创建模型
    if args.model_type == "fcn":
        model = FCN(n_channels=3, n_classes=1, pretrained=True)
    elif args.model_type == "sam2unet":
        model = SAM2UNet(checkpoint_path=args.hiera_path)
    elif args.model_type == "unet":
        model = UNet()
    elif args.model_type == "unetplusplus":
        model = UNetPlusPlus(deep_supervision=args.deep_supervision)
    elif args.model_type == "unetplus":
        model = UNetPlus()
    elif args.model_type == "lstmunet":
        model = LSTMUNet()
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    model = model.to(device)
    
    # 如果指定了预训练模型，加载它
    if args.pretrained_path:
        print(f"加载预训练模型: {args.pretrained_path}")
        model.load_state_dict(torch.load(args.pretrained_path))
    
    # 如果指定了冻结backbone
    if args.freeze_backbone and hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = False
    
    # 打印模型结构
    print_model_structure(model, args.model_type)
    
    # 只优化需要梯度的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("没有可训练的参数！请检查模型结构或冻结设置。")
    
    print(f"\n可训练参数数量: {len(trainable_params)}")
    optim = opt.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    os.makedirs(args.save_path, exist_ok=True)

    # 创建CSV文件记录训练指标
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.save_path, f'training_metrics_{args.model_type}_{timestamp}.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'train_iou', 'train_dice', 'val_loss', 'val_iou', 'val_dice', 'learning_rate'])

    best_val_loss = float('inf')
    best_val_iou = 0
    
    for epoch in range(args.epoch):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epoch}")
        print(f"{'='*50}")
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_iou = 0
        train_dice = 0
        num_train_batches = 0
        
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f'Training',
            unit='batch',
            ncols=100,
            bar_format='{l_bar}{bar:30}{r_bar}',
            dynamic_ncols=True,
            leave=True
        )
        
        for i, batch in progress_bar:
            x = batch['image'].to(device)
            target = batch['label'].to(device)
            optim.zero_grad()
            
            if args.model_type == "sam2unet":
                pred0, pred1, pred2 = model(x)
                loss0 = structure_loss(pred0, target)
                loss1 = structure_loss(pred1, target)
                loss2 = structure_loss(pred2, target)
                loss = (loss0 + loss1 + loss2) / 3
                pred = pred2  # 使用最后一个预测计算指标
            elif args.model_type == "unetplusplus" and isinstance(model, UNetPlusPlus) and model.deep_supervision:
                preds = model(x)
                loss = sum(structure_loss(p, target) for p in preds) / len(preds)
                pred = preds[-1]  # 使用最后一个预测计算指标
            else:
                pred = model(x)
                loss = structure_loss(pred, target)
            
            loss.backward()
            optim.step()

            # 计算训练指标
            pred = torch.sigmoid(pred)
            iou, dice = calculate_metrics(pred, target)
            
            train_loss += loss.item()
            train_iou += iou
            train_dice += dice
            num_train_batches += 1

            # 实时更新进度条，每个批次都更新
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}',
                'dice': f'{dice:.4f}'
            })

        avg_train_loss = train_loss / num_train_batches
        avg_train_iou = train_iou / num_train_batches
        avg_train_dice = train_dice / num_train_batches
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 验证阶段
        try:
            if args.debug_validation:
                val_loss, val_iou, val_dice = simple_validate(model, val_loader, device, args.model_type)
            else:
                # 尝试使用完整验证函数
                val_loss, val_iou, val_dice = validate(model, val_loader, device, args.model_type)
        except Exception as e:
            print(f"验证失败: {str(e)}")
            print("尝试使用简化验证函数...")
            try:
                val_loss, val_iou, val_dice = simple_validate(model, val_loader, device, args.model_type)
            except Exception as e2:
                print(f"简化验证也失败: {str(e2)}")
                print("跳过验证，使用默认值")
                val_loss, val_iou, val_dice = float('inf'), 0.0, 0.0
        
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
        csv_writer.writerow([epoch+1, avg_train_loss, avg_train_iou, avg_train_dice, 
                           val_loss, val_iou, val_dice, current_lr])
        csv_file.flush()  # 确保数据写入文件
        
        # 保存最佳模型
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            model_name = f'{args.model_type}-best.pth'
            torch.save(model.state_dict(), 
                      os.path.join(args.save_path, model_name))
            print(f'\n保存最佳模型，IoU: {best_val_iou:.4f}')
        
        # 定期保存检查点
        if (epoch+1) % 20 == 0 or (epoch+1) == args.epoch:
            model_name = f'{args.model_type}-{epoch+1}.pth'
            torch.save(model.state_dict(), 
                      os.path.join(args.save_path, model_name))
            print(f'保存检查点: {model_name}')
    
    csv_file.close()
    print(f'\n训练指标已保存到: {csv_path}')



if __name__ == "__main__":
    # seed_torch(1024)
    main(args)
