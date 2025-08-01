#!/usr/bin/env python3
# Train/Fine Tune SAM 2 on LabPics 1 dataset
# This mode use several images in a single batch
# Labpics can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import csv
from datetime import datetime
import random
import torch.optim as opt
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
import re
import sys
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 导入dataset.py中的FullDataset
from dataset import FullDataset

def parse_shell_args(shell_file='train_sam2.sh'):
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
parser = argparse.ArgumentParser("SAM2 Model Training")
parser.add_argument("--sam2_checkpoint", type=str, default="/home/SAM2-UNet/sam2_hiera_large.pt",
                    help="path to SAM2 model weight")
parser.add_argument("--model_cfg", type=str, default="sam2_hiera_l.yaml",
                    help="SAM2 model config")
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
parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--weight_decay", default=4e-5, type=float)
parser.add_argument("--freeze_backbone", action="store_true",
                    help="是否冻结主干网络参数")
parser.add_argument("--num_workers", type=int, default=3,
                    help="数据加载时使用的子进程数量")

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
    pred = (pred > 0.5).float()
    intersection = (pred * mask).sum()
    union = (pred + mask).sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred.sum() + mask.sum() + 1e-6)
    return iou.item(), dice.item()

def read_single(dataset, max_retries=10): 
    """读取随机图像和单个掩码，添加重试次数限制"""
    for retry in range(max_retries):
        try:
            # 随机选择一个数据样本
            idx = np.random.randint(len(dataset))
            data = dataset[idx]
            
            # 获取图像和标注
            image = data['image'].numpy().transpose(1, 2, 0)  # CHW -> HWC
            ann_map = data['label'].numpy()  # 已经是单通道
            
            # 处理标注形状 - 如果是3D，取第一个通道
            if len(ann_map.shape) == 3:
                ann_map = ann_map[0]  # 取第一个通道
            
            # 反归一化图像
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image * std + mean
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            
            # resize image to 1024x1024
            r = np.min([1024 / image.shape[1], 1024 / image.shape[0]]) # scaling factor
            image = cv2.resize(image, (int(image.shape[1] * r), int(image.shape[0] * r)))
            ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
            
            # 填充到1024x1024
            if image.shape[0] < 1024:
                image = np.concatenate([image, np.zeros([1024 - image.shape[0], image.shape[1], 3], dtype=np.uint8)], axis=0)
                ann_map = np.concatenate([ann_map, np.zeros([1024 - ann_map.shape[0], ann_map.shape[1]], dtype=np.float32)], axis=0)
            if image.shape[1] < 1024:
                image = np.concatenate([image, np.zeros([image.shape[0], 1024 - image.shape[1], 3], dtype=np.uint8)], axis=1)
                ann_map = np.concatenate([ann_map, np.zeros([ann_map.shape[0], 1024 - ann_map.shape[1]], dtype=np.float32)], axis=1)

            # 处理标注 - 适配单通道二值标注
            if len(ann_map.shape) == 2:
                # 单通道标注，直接使用
                mask = (ann_map > 0.5).astype(np.uint8)  # 二值化
            else:
                # 多通道标注，使用原来的逻辑
                ann_map = np.stack([ann_map, ann_map, ann_map], axis=-1) if len(ann_map.shape) == 2 else ann_map
                # merge vessels and materials annotations
                mat_map = ann_map[:,:,0] # material annotation map
                ves_map = ann_map[:,:,2] if ann_map.shape[2] > 2 else ann_map[:,:,0] # vessel annotation map
                mat_map[mat_map==0] = ves_map[mat_map==0]*(mat_map.max()+1) # merge maps
                
                # Get binary masks and points
                inds = np.unique(mat_map)[1:] # load all indices
                if len(inds) > 0:
                    ind = inds[np.random.randint(len(inds))]  # pick single segment
                    mask = (mat_map == ind).astype(np.uint8) # make binary mask corresponding to index ind
                else:
                    print(f"警告: 样本 {idx} 没有有效标注，重试 {retry+1}/{max_retries}")
                    continue

            # 检查掩码是否有效
            coords = np.argwhere(mask > 0) # get all coordinates in mask
            if len(coords) == 0:
                print(f"警告: 样本 {idx} 掩码为空，重试 {retry+1}/{max_retries}")
                continue
            
            # 选择随机点
            yx = np.array(coords[np.random.randint(len(coords))]) # choose random point/coordinate
            return image, mask, [[yx[1], yx[0]]]
            
        except Exception as e:
            print(f"警告: 处理样本 {idx} 时出错: {str(e)}，重试 {retry+1}/{max_retries}")
            continue
    
    # 如果所有重试都失败，返回默认值
    print(f"错误: 在 {max_retries} 次重试后仍无法读取有效样本")
    return np.zeros((1024, 1024, 3), dtype=np.uint8), np.zeros((1024, 1024), dtype=np.uint8), [[512, 512]]

def read_batch(dataset, batch_size=4):
    """读取一个批次的数据"""
    limage = []
    lmask = []
    linput_point = []
    
    for i in range(batch_size):
        try:
            image, mask, input_point = read_single(dataset)
            limage.append(image)
            lmask.append(mask)
            linput_point.append(input_point)
        except Exception as e:
            print(f"警告: 读取批次样本 {i} 时出错: {str(e)}")
            # 添加默认值
            limage.append(np.zeros((1024, 1024, 3), dtype=np.uint8))
            lmask.append(np.zeros((1024, 1024), dtype=np.uint8))
            linput_point.append([[512, 512]])

    return limage, np.array(lmask), np.array(linput_point), np.ones([batch_size, 1])

@torch.no_grad()
def validate(predictor, val_dataset, device, num_val_samples=100):
    """验证函数"""
    predictor.model.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0
    num_samples = 0
    
    try:
        progress_bar = tqdm(
            range(num_val_samples),
            desc='Validating',
            unit='sample',
            ncols=100,
            bar_format='{l_bar}{bar:30}{r_bar}',
            dynamic_ncols=True,
            leave=True
        )
        
        for _ in progress_bar:
            try:
                image, mask, input_point, input_label = read_batch(val_dataset, batch_size=1)
                
                if mask.shape[0] == 0:
                    continue

                predictor.set_image_batch(image)
                # prompt encoding
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None, masks=None)

                # mask decoder
                high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"], 
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=False,
                    high_res_features=high_res_features,
                )
                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

                # Loss calculation
                gt_mask = torch.tensor(mask.astype(np.float32)).to(device)
                prd_mask = torch.sigmoid(prd_masks[:, 0])
                
                # 使用结构损失
                loss = structure_loss(prd_masks[:, 0], gt_mask)
                
                # 计算指标
                iou, dice = calculate_metrics(prd_mask, gt_mask)
                
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
                print(f"处理验证样本时出错: {str(e)}")
                continue
        
        if num_samples == 0:
            raise ValueError("验证过程中没有成功处理任何样本")
            
        return total_loss/num_samples, total_iou/num_samples, total_dice/num_samples
        
    except Exception as e:
        print(f"验证过程中出错: {str(e)}")
        return float('inf'), 0.0, 0.0

def print_model_structure(predictor):
    """打印模型结构"""
    print(f"\n{'='*50}")
    print(f"模型类型: SAM2")
    print(f"{'='*50}")
    
    # 计算模型总参数量
    total_params = sum(p.numel() for p in predictor.model.parameters())
    trainable_params = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
    
    print(f"\n模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"不可训练参数量: {total_params - trainable_params:,}")
    
    # 分别统计各个组件的参数量
    if hasattr(predictor.model, 'image_encoder'):
        img_encoder_params = sum(p.numel() for p in predictor.model.image_encoder.parameters())
        img_encoder_trainable = sum(p.numel() for p in predictor.model.image_encoder.parameters() if p.requires_grad)
        print(f"图像编码器参数: {img_encoder_params:,} (可训练: {img_encoder_trainable:,})")
    
    if hasattr(predictor.model, 'sam_prompt_encoder'):
        prompt_encoder_params = sum(p.numel() for p in predictor.model.sam_prompt_encoder.parameters())
        prompt_encoder_trainable = sum(p.numel() for p in predictor.model.sam_prompt_encoder.parameters() if p.requires_grad)
        print(f"提示编码器参数: {prompt_encoder_params:,} (可训练: {prompt_encoder_trainable:,})")
    
    if hasattr(predictor.model, 'sam_mask_decoder'):
        mask_decoder_params = sum(p.numel() for p in predictor.model.sam_mask_decoder.parameters())
        mask_decoder_trainable = sum(p.numel() for p in predictor.model.sam_mask_decoder.parameters() if p.requires_grad)
        print(f"掩码解码器参数: {mask_decoder_params:,} (可训练: {mask_decoder_trainable:,})")
    
    print(f"{'='*50}")
    print("模型组件:")
    print(f"  - 图像编码器: {'✓' if hasattr(predictor.model, 'image_encoder') else '✗'}")
    print(f"  - 提示编码器: {'✓' if hasattr(predictor.model, 'sam_prompt_encoder') else '✗'}")
    print(f"  - 掩码解码器: {'✓' if hasattr(predictor.model, 'sam_mask_decoder') else '✗'}")
    print(f"{'='*50}\n")

def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建训练和验证数据集
    print("正在加载数据集...")
    train_dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, mode='train')
    val_dataset = FullDataset(args.val_image_path, args.val_mask_path, 352, mode='val')
    
    print(f"训练数据集大小: {len(train_dataset)}")
    print(f"验证数据集大小: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        raise ValueError("训练数据集为空！请检查数据路径。")
    if len(val_dataset) == 0:
        raise ValueError("验证数据集为空！请检查数据路径。")
    
    # 测试数据加载
    print("测试数据加载...")
    try:
        test_data = train_dataset[0]
        print(f"数据加载测试通过")
    except Exception as e:
        print(f"数据加载测试失败: {str(e)}")
        raise
    
    # 加载SAM2模型
    print("正在加载SAM2模型...")
    try:
        sam2_model = build_sam2(args.model_cfg, args.sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        print("SAM2模型加载成功")
    except Exception as e:
        print(f"SAM2模型加载失败: {str(e)}")
        raise
    
    # 设置训练参数
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)
    predictor.model.image_encoder.train(True)
    
    # 如果指定了冻结backbone
    if args.freeze_backbone:
        for param in predictor.model.image_encoder.parameters():
            param.requires_grad = False
        print("已冻结图像编码器参数")
    
    # 打印模型结构
    print_model_structure(predictor)
    
    # 只优化需要梯度的参数
    trainable_params = [p for p in predictor.model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("没有可训练的参数！请检查模型结构或冻结设置。")
    
    print(f"\n可训练参数数量: {len(trainable_params)}")
    
    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=1.0e-7)
    scaler = torch.cuda.amp.GradScaler()  # 混合精度
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)

    # 创建CSV文件记录训练指标
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.save_path, f'sam2_training_metrics_{timestamp}.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'train_iou', 'train_dice', 'val_loss', 'val_iou', 'val_dice', 'learning_rate'])

    best_val_loss = float('inf')
    best_val_iou = 0
    
    # 训练循环
    for epoch in range(args.epoch):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epoch}")
        print(f"{'='*50}")
        
        # 训练阶段
        predictor.model.train()
        train_loss = 0
        train_iou = 0
        train_dice = 0
        num_train_batches = 0
        
        # 计算每个epoch的批次数 - 修复计算逻辑
        num_batches = max(1, len(train_dataset) // args.batch_size)
        
        progress_bar = tqdm(
            range(num_batches),
            desc=f'Training Epoch {epoch+1}',
            unit='batch',
            ncols=100,
            bar_format='{l_bar}{bar:30}{r_bar}',
            dynamic_ncols=True,
            leave=True
        )
        
        for i in progress_bar:
            try:
                with torch.cuda.amp.autocast():
                    image, mask, input_point, input_label = read_batch(train_dataset, batch_size=args.batch_size)

                    if mask.shape[0] == 0:
                        continue

                    predictor.set_image_batch(image)
                    
                    # prompt encoding
                    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
                    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None, masks=None)

                    # mask decoder
                    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
                    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                        image_embeddings=predictor._features["image_embed"], 
                        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                        repeat_image=False,
                        high_res_features=high_res_features,
                    )
                    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

                    # Loss calculation
                    gt_mask = torch.tensor(mask.astype(np.float32)).to(device)
                    prd_mask = torch.sigmoid(prd_masks[:, 0])
                    
                    # 使用结构损失
                    loss = structure_loss(prd_masks[:, 0], gt_mask)
                    
                    # 计算指标
                    iou, dice = calculate_metrics(prd_mask, gt_mask)

                # 反向传播
                predictor.model.zero_grad()
                scaler.scale(loss).backward()
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
                
                # 强制更新进度条
                progress_bar.refresh()
                    
            except Exception as e:
                print(f"处理训练批次 {i} 时出错: {str(e)}")
                continue

        avg_train_loss = train_loss / num_train_batches if num_train_batches > 0 else 0
        avg_train_iou = train_iou / num_train_batches if num_train_batches > 0 else 0
        avg_train_dice = train_dice / num_train_batches if num_train_batches > 0 else 0
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 验证阶段
        val_loss, val_iou, val_dice = validate(predictor, val_dataset, device)
        
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
            model_name = f'sam2-best.pth'
            torch.save(predictor.model.state_dict(), 
                    os.path.join(args.save_path, model_name))
            print(f'\n保存最佳模型，IoU: {best_val_iou:.4f}')
        
        # 定期保存检查点
        if (epoch+1) % 20 == 0 or (epoch+1) == args.epoch:
            model_name = f'sam2-{epoch+1}.pth'
            torch.save(predictor.model.state_dict(), 
                    os.path.join(args.save_path, model_name))
            print(f'保存检查点: {model_name}')
    
    csv_file.close()
    print(f'\n训练指标已保存到: {csv_path}')

if __name__ == "__main__":
    main(args)