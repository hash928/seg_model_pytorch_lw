import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from dataset import FullDataset
import argparse
import csv
from datetime import datetime
import re
import sys
import style.logo

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

def calculate_metrics(pred, mask):
    """计算IoU和Dice指标"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * mask).sum()
    union = (pred + mask).sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred.sum() + mask.sum() + 1e-6)
    return iou.item(), dice.item()

def get_center_point(mask):
    """从掩码中获取中心点坐标"""
    # 找到掩码中非零像素的坐标
    coords = torch.nonzero(mask.squeeze())
    if len(coords) == 0:
        # 如果掩码为空，返回图像中心
        h, w = mask.shape[-2:]
        return [w//2, h//2]
    
    # 计算质心
    center_y = coords[:, 0].float().mean().item()
    center_x = coords[:, 1].float().mean().item()
    return [int(center_x), int(center_y)]

def process_batch_sam1(sam, images, masks, device):
    """处理SAM1批次数据，逐个处理每个样本"""
    batch_size = images.shape[0]
    all_masks = []
    all_losses = []
    all_adjusted_masks = []  # 存储调整后的掩码
    
    for i in range(batch_size):
        # 处理单个样本
        image = images[i:i+1]  # 保持批次维度
        mask = masks[i:i+1]
        
        # 获取图像尺寸
        _, _, h, w = image.shape
        
        # 从掩码中获取合适的提示点
        center_point = get_center_point(mask)
        input_points = torch.tensor([[[center_point[0], center_point[1]]]], 
                                  dtype=torch.float, device=device)
        input_labels = torch.ones((1, 1), dtype=torch.int, device=device)
        
        # 图像编码
        with torch.no_grad():
            image_embeddings = sam.image_encoder(image)
        
        # 提示编码
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=(input_points, input_labels), boxes=None, masks=None
        )
        
        # 掩码解码
        low_res_masks, iou_predictions = sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        # 确保掩码尺寸匹配
        adjusted_mask = mask
        if low_res_masks.shape[-2:] != mask.shape[-2:]:
            adjusted_mask = F.interpolate(mask, size=low_res_masks.shape[-2:], mode='nearest')
        
        # 计算损失
        loss = structure_loss(low_res_masks, adjusted_mask)
        
        all_masks.append(low_res_masks)
        all_adjusted_masks.append(adjusted_mask)
        all_losses.append(loss)
    
    # 合并结果
    combined_masks = torch.cat(all_masks, dim=0)
    combined_adjusted_masks = torch.cat(all_adjusted_masks, dim=0)
    combined_loss = torch.stack(all_losses).mean()
    
    return combined_masks, combined_adjusted_masks, combined_loss

def print_model_structure(sam):
    """打印模型结构"""
    print(f"\n{'='*50}")
    print(f"模型类型: SAM1")
    print(f"{'='*50}")
    
    # 计算模型总参数量
    total_params = sum(p.numel() for p in sam.parameters())
    trainable_params = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    
    print(f"\n模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"不可训练参数量: {total_params - trainable_params:,}")
    
    # 分别统计各个组件的参数量
    if hasattr(sam, 'image_encoder'):
        img_encoder_params = sum(p.numel() for p in sam.image_encoder.parameters())
        img_encoder_trainable = sum(p.numel() for p in sam.image_encoder.parameters() if p.requires_grad)
        print(f"图像编码器参数: {img_encoder_params:,} (可训练: {img_encoder_trainable:,})")
    
    if hasattr(sam, 'prompt_encoder'):
        prompt_encoder_params = sum(p.numel() for p in sam.prompt_encoder.parameters())
        prompt_encoder_trainable = sum(p.numel() for p in sam.prompt_encoder.parameters() if p.requires_grad)
        print(f"提示编码器参数: {prompt_encoder_params:,} (可训练: {prompt_encoder_trainable:,})")
    
    if hasattr(sam, 'mask_decoder'):
        mask_decoder_params = sum(p.numel() for p in sam.mask_decoder.parameters())
        mask_decoder_trainable = sum(p.numel() for p in sam.mask_decoder.parameters() if p.requires_grad)
        print(f"掩码解码器参数: {mask_decoder_params:,} (可训练: {mask_decoder_trainable:,})")
    
    print(f"{'='*50}")
    print("模型组件:")
    print(f"  - 图像编码器: {'✓' if hasattr(sam, 'image_encoder') else '✗'}")
    print(f"  - 提示编码器: {'✓' if hasattr(sam, 'prompt_encoder') else '✗'}")
    print(f"  - 掩码解码器: {'✓' if hasattr(sam, 'mask_decoder') else '✗'}")
    print(f"{'='*50}\n")

def train_sam1_fixed(
    model_type="vit_b",
    checkpoint="sam_vit_b_01ec64.pth",
    train_image_dir="train/images/",
    train_mask_dir="train/masks/",
    val_image_dir="val/images/",
    val_mask_dir="val/masks/",
    save_path="sam1_finetune.pth",
    batch_size=4,  # 现在支持更大的batch_size
    lr=1e-5,
    weight_decay=1e-4,  # 添加权重衰减参数
    num_epochs=20,
    device="cuda",
    num_workers=2,
    freeze_backbone=False
):
    # 设置设备
    print(f"使用设备: {device}")
    
    # 根据模型类型设置合适的图像尺寸
    image_size = 1024
    
    # 使用FullDataset
    print("正在加载数据集...")
    train_dataset = FullDataset(train_image_dir, train_mask_dir, image_size, mode='train')
    val_dataset = FullDataset(val_image_dir, val_mask_dir, image_size, mode='val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    print(f"训练数据集大小: {len(train_dataset)}")
    print(f"验证数据集大小: {len(val_dataset)}")
    print(f"图像尺寸: {image_size}x{image_size}")
    print(f"批次大小: {batch_size}")
    
    if len(train_dataset) == 0:
        raise ValueError("训练数据集为空！请检查数据路径。")
    if len(val_dataset) == 0:
        raise ValueError("验证数据集为空！请检查数据路径。")

    # 加载SAM1模型
    print("正在加载SAM1模型...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    sam.train()
    print("SAM1模型加载成功")
    
    # 验证SAM1模型
    print("\n验证SAM1模型...")
    total_params = sum(p.numel() for p in sam.parameters())
    print(f"总参数量: {total_params:,}")
    
    # 检查关键组件
    required_components = ['image_encoder', 'prompt_encoder', 'mask_decoder']
    for component in required_components:
        if hasattr(sam, component):
            print(f"✅ {component}: 存在")
        else:
            print(f"❌ {component}: 缺失")
            raise ValueError(f"SAM1模型缺少必要组件: {component}")
    
    # 验证参数量范围
    if model_type == "vit_b" and 90_000_000 < total_params < 100_000_000:
        print("✅ 参数量符合SAM1 ViT-B模型")
    elif model_type == "vit_l" and 300_000_000 < total_params < 400_000_000:
        print("✅ 参数量符合SAM1 ViT-L模型")
    elif model_type == "vit_h" and 600_000_000 < total_params < 700_000_000:
        print("✅ 参数量符合SAM1 ViT-H模型")
    else:
        print(f"⚠️  参数量 ({total_params:,}) 与预期不符，请确认这是SAM1 {model_type}模型")
    
    # 检查模型类名
    model_class_name = sam.__class__.__name__
    print(f"模型类名: {model_class_name}")
    if "Sam" in model_class_name:
        print("✅ 模型类名符合SAM特征")
    else:
        print("⚠️  模型类名不符合SAM特征")
    
    print("SAM1模型验证完成！")
    
    # 设置优化器
    optimizer = optim.AdamW(sam.mask_decoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, num_epochs, eta_min=1.0e-7)
    scaler = torch.amp.GradScaler('cuda')  # 混合精度 - 修复弃用警告

    if freeze_backbone:
        for param in sam.image_encoder.parameters():
            param.requires_grad = False
        print("已冻结图像编码器参数")

    # 打印模型结构
    print_model_structure(sam)
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    print(f"模型和训练指标将保存到: {save_path}")

    # 创建CSV文件记录训练指标
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(save_path, f'sam1_training_metrics_{timestamp}.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'train_iou', 'train_dice', 'val_loss', 'val_iou', 'val_dice', 'learning_rate'])

    best_val_loss = float('inf')
    best_val_iou = 0

    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        
        # 训练阶段
        sam.train()
        train_loss = 0
        train_iou = 0
        train_dice = 0
        num_train_batches = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f'Training Epoch {epoch+1}',
            unit='batch',
            ncols=100,
            bar_format='{l_bar}{bar:30}{r_bar}',
            dynamic_ncols=True,
            leave=True
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                with torch.amp.autocast('cuda'):
                    images = batch['image'].to(device)
                    masks = batch['label'].to(device)
                    
                    # 确保图像尺寸正确
                    b, _, h, w = images.shape
                    if h != image_size or w != image_size:
                        print(f"警告：批次 {batch_idx} 的图像尺寸为 {h}x{w}，期望 {image_size}x{image_size}")
                        continue
                    
                    # 检查数据有效性
                    if torch.isnan(images).any() or torch.isnan(masks).any():
                        print(f"警告：批次 {batch_idx} 包含NaN值，跳过")
                        continue
                    
                    # 使用修复的批次处理函数
                    low_res_masks, adjusted_masks, loss = process_batch_sam1(sam, images, masks, device)
                    
                    # 检查预测结果的有效性
                    if torch.isnan(low_res_masks).any():
                        continue
                    
                    # 检查损失值
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    
                    # 计算指标
                    iou, dice = calculate_metrics(low_res_masks, adjusted_masks)

                # 反向传播
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # 检查梯度
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告：批次 {batch_idx} 的损失值异常，跳过梯度更新")
                    continue
                
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                train_iou += iou
                train_dice += dice
                num_train_batches += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'iou': f'{iou:.4f}',
                    'dice': f'{dice:.4f}'
                })
                
            except Exception as e:
                print(f"处理训练批次 {batch_idx} 时出错: {str(e)}")
                print(f"图像形状: {images.shape if 'images' in locals() else 'N/A'}")
                print(f"掩码形状: {masks.shape if 'masks' in locals() else 'N/A'}")
                continue

        avg_train_loss = train_loss / num_train_batches if num_train_batches > 0 else 0
        avg_train_iou = train_iou / num_train_batches if num_train_batches > 0 else 0
        avg_train_dice = train_dice / num_train_batches if num_train_batches > 0 else 0

        # 验证阶段
        sam.eval()
        val_loss = 0
        val_iou = 0
        val_dice = 0
        num_val_batches = 0
        
        with torch.no_grad():
            val_progress_bar = tqdm(
                val_loader,
                desc='Validating',
                unit='batch',
                ncols=100,
                bar_format='{l_bar}{bar:30}{r_bar}',
                dynamic_ncols=True,
                leave=True
            )
            
            for batch_idx, batch in enumerate(val_progress_bar):
                try:
                    images = batch['image'].to(device)
                    masks = batch['label'].to(device)
                    
                    # 确保图像尺寸正确
                    b, _, h, w = images.shape
                    if h != image_size or w != image_size:
                        continue
                    
                    # 检查数据有效性
                    if torch.isnan(images).any() or torch.isnan(masks).any():
                        continue
                    
                    # 使用修复的批次处理函数
                    low_res_masks, adjusted_masks, loss = process_batch_sam1(sam, images, masks, device)
                    
                    # 检查预测结果的有效性
                    if torch.isnan(low_res_masks).any():
                        continue
                    
                    # 检查损失值
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    
                    # 计算指标
                    iou, dice = calculate_metrics(low_res_masks, adjusted_masks)
                    
                    val_loss += loss.item()
                    val_iou += iou
                    val_dice += dice
                    num_val_batches += 1
                    
                    # 更新进度条
                    val_progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'iou': f'{iou:.4f}',
                        'dice': f'{dice:.4f}'
                    })
                    
                except Exception as e:
                    print(f"处理验证批次 {batch_idx} 时出错: {str(e)}")
                    print(f"图像形状: {images.shape if 'images' in locals() else 'N/A'}")
                    print(f"掩码形状: {masks.shape if 'masks' in locals() else 'N/A'}")
                    continue

        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        avg_val_iou = val_iou / num_val_batches if num_val_batches > 0 else 0
        avg_val_dice = val_dice / num_val_batches if num_val_batches > 0 else 0
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 打印训练和验证指标
        print(f"\n训练指标:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Train IoU: {avg_train_iou:.4f}")
        print(f"Train Dice: {avg_train_dice:.4f}")
        print(f"\n验证指标:")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val IoU: {avg_val_iou:.4f}")
        print(f"Val Dice: {avg_val_dice:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # 记录指标到CSV
        csv_writer.writerow([epoch+1, avg_train_loss, avg_train_iou, avg_train_dice, 
                        avg_val_loss, avg_val_iou, avg_val_dice, current_lr])
        csv_file.flush()  # 确保数据写入文件
        
        # 保存最佳模型
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            best_model_path = os.path.join(save_path, 'sam1-best.pth')
            torch.save(sam.state_dict(), best_model_path)
            print(f'\n保存最佳模型到: {best_model_path}')
            print(f'最佳IoU: {best_val_iou:.4f}')
        
        # 定期保存检查点
        if (epoch+1) % 20 == 0 or (epoch+1) == num_epochs:
            checkpoint_path = os.path.join(save_path, f'sam1-epoch-{epoch+1}.pth')
            torch.save(sam.state_dict(), checkpoint_path)
            print(f'保存检查点到: {checkpoint_path}')

    # 保存最终模型到指定路径
    final_model_path = os.path.join(save_path, "sam1_final.pth")
    torch.save(sam.state_dict(), final_model_path)

    csv_file.close()
    print(f'\n训练完成！')
    print(f"训练日志目录: {save_path}")
    print(f"训练指标文件: {csv_path}")
    print(f"最佳模型: {os.path.join(save_path, 'sam1-best.pth')}")
    print(f"最终模型: {final_model_path}")

def parse_shell_args(shell_file='train_sam1.sh'):
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

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser("SAM1 Model Training (Fixed Version)")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="path to SAM1 model weight")
    parser.add_argument("--model_type", type=str, default="vit_b", help="SAM1 model type (vit_b/vit_l/vit_h)")
    parser.add_argument("--train_image_path", type=str, required=True, help="path to training images")
    parser.add_argument("--train_mask_path", type=str, required=True, help="path to training masks")
    parser.add_argument("--val_image_path", type=str, required=True, help="path to validation images")
    parser.add_argument("--val_mask_path", type=str, required=True, help="path to validation masks")
    parser.add_argument("--save_path", type=str, required=True, help="path to save the checkpoint")
    parser.add_argument("--epoch", type=int, default=20, help="training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size (now supports larger values)")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--freeze_backbone", action="store_true", help="freeze image encoder backbone")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay for optimizer")

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

    train_sam1_fixed(
        model_type=args.model_type,
        checkpoint=args.sam_checkpoint,
        train_image_dir=args.train_image_path,
        train_mask_dir=args.train_mask_path,
        val_image_dir=args.val_image_path,
        val_mask_dir=args.val_mask_path,
        save_path=args.save_path,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.epoch,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=args.num_workers,
        freeze_backbone=args.freeze_backbone,
        weight_decay=args.weight_decay
    ) 