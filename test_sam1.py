#!/usr/bin/env python3
# SAM1 模型测试脚本

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime
import random
import re
import sys

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from segment_anything import sam_model_registry, SamPredictor
from dataset import FullDataset

def parse_shell_args(shell_file='test_sam1.sh'):
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
        # 匹配 --xxx "yyy" 或 --xxx
        pattern = r'--(\w+)(?:\s+"([^"]+)")?'
        matches = re.findall(pattern, content)
        for key, value in matches:
            # 处理布尔参数
            if value == '':
                args_dict[key] = True
            else:
                # 转换数值类型
                if value is not None and value.isdigit():
                    value = int(value)
                elif value is not None and re.match(r'^-?\d*\.\d+$', value):
                    value = float(value)
                args_dict[key] = value
        print("从shell脚本读取的参数：")
        for key, value in args_dict.items():
            print(f"{key}: {value}")
    except FileNotFoundError:
        print(f"警告：未找到shell脚本 {shell_file}，将使用命令行参数")
    except Exception as e:
        print(f"警告：解析shell脚本时出错：{e}，将使用命令行参数")
    return args_dict

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser("SAM1 Model Testing")
    parser.add_argument("--model_path", type=str, required=True,
                        help="训练好的SAM1模型路径")
    parser.add_argument("--model_type", type=str, default="vit_b",
                        help="SAM1模型类型 (vit_b/vit_l/vit_h)")
    parser.add_argument("--test_image_path", type=str, required=True,
                        help="测试图像路径")
    parser.add_argument("--test_mask_path", type=str, required=True,
                        help="测试标注路径")
    parser.add_argument("--save_path", type=str, default="./test_results",
                        help="测试结果保存路径")
    parser.add_argument("--visualize", action="store_true",
                        help="是否保存可视化结果")
    
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
    
    return parser.parse_args()

def calculate_metrics(pred, mask):
    """计算IoU和Dice指标"""
    # 确保输入是2D张量
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    
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

def read_single_for_test(dataset, idx):
    """读取单个测试样本"""
    data = dataset[idx]
    
    # 获取原始标签路径
    original_mask_path = dataset.gts[idx]
    mask_name = os.path.basename(original_mask_path)
    
    # 获取图像和标注
    image = data['image'].detach().numpy().transpose(1, 2, 0)  # CHW -> HWC
    ann_map = data['label'].detach().numpy()  # 已经是单通道
    
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
            ind = inds[0]  # 取第一个segment
            mask = (mat_map == ind).astype(np.uint8) # make binary mask corresponding to index ind
        else:
            mask = np.zeros((1024, 1024), dtype=np.uint8)

    # 选择随机点
    coords = np.argwhere(mask > 0) # get all coordinates in mask
    if len(coords) == 0:
        # 如果没有有效点，选择图像中心
        yx = np.array([512, 512])
    else:
        yx = np.array(coords[np.random.randint(len(coords))]) # choose random point/coordinate
    
    return image, mask, [[yx[1], yx[0]]], original_mask_path, mask_name

def test_single_image(predictor, image, mask, input_point, device, save_path=None, sample_idx=0, original_mask_path=None, mask_name=None):
    """测试单张图像"""
    try:
        # 设置图像
        predictor.set_image(image)
        
        # 准备输入 - 修复输入点格式
        # input_point 是 [[x, y]] 格式，需要转换为正确的numpy数组
        if isinstance(input_point, list) and len(input_point) > 0:
            input_points = np.array(input_point, dtype=np.float32)  # [[x, y]]
        else:
            input_points = np.array([[512, 512]], dtype=np.float32)  # 默认中心点
            
        input_labels = np.array([1], dtype=np.int32)
        
        print(f"样本 {sample_idx}: 输入点形状: {input_points.shape}, 标签形状: {input_labels.shape}")
        print(f"样本 {sample_idx}: 输入点内容: {input_points}, 标签内容: {input_labels}")
        
        # 预测掩码
        try:
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
            )
            print(f"样本 {sample_idx}: 预测成功，masks形状: {masks.shape}")
        except Exception as predict_error:
            print(f"样本 {sample_idx}: 预测失败: {str(predict_error)}")
            return None
        
        # 获取预测结果
        pred_mask = masks[0]  # 取第一个掩码
        
        # 调试信息：打印原始形状
        print(f"样本 {sample_idx}: 原始pred_mask形状: {pred_mask.shape}, mask形状: {mask.shape}")
        
        # 更安全的维度处理 - 直接取最后两个维度
        if len(pred_mask.shape) > 2:
            # 如果是多维，取最后两个维度
            pred_mask = pred_mask.reshape(-1, pred_mask.shape[-2], pred_mask.shape[-1])[0]
        
        if len(mask.shape) > 2:
            # 如果是多维，取最后两个维度
            mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1])[0]
        
        print(f"样本 {sample_idx}: 处理后pred_mask形状: {pred_mask.shape}, mask形状: {mask.shape}")
        
        # 转换为张量
        gt_mask = torch.tensor(mask.astype(np.float32)).to(device)
        pred_tensor = torch.tensor(pred_mask.astype(np.float32)).to(device)
        
        print(f"样本 {sample_idx}: 最终gt_mask形状: {gt_mask.shape}, pred_tensor形状: {pred_tensor.shape}")
        
        # 计算指标
        iou, dice = calculate_metrics(pred_tensor, gt_mask)
        
        # 可视化结果
        if save_path:
            visualize_result(image, mask, pred_mask, input_point, 
                           iou, dice, save_path, sample_idx, original_mask_path, mask_name)
        
        return {
            'iou': iou,
            'dice': dice,
            'prediction': pred_mask,
            'gt_mask': mask,
            'input_point': input_point,
            'mask_name': mask_name
        }
        
    except Exception as e:
        print(f"测试样本 {sample_idx} 时出错: {str(e)}")
        print(f"错误详情 - pred_mask形状: {pred_mask.shape if 'pred_mask' in locals() else 'N/A'}")
        print(f"错误详情 - mask形状: {mask.shape if 'mask' in locals() else 'N/A'}")
        print(f"错误详情 - image形状: {image.shape if 'image' in locals() else 'N/A'}")
        print(f"错误详情 - input_point: {input_point if 'input_point' in locals() else 'N/A'}")
        return None

def visualize_result(image, gt_mask, pred_mask, input_point, iou, dice, save_path, sample_idx, original_mask_path=None, mask_name=None):
    """可视化测试结果"""
    plt.figure(figsize=(15, 5))
    
    # 确保图像数据格式正确
    if len(image.shape) == 3 and image.shape[0] == 1:
        image = image.squeeze(0)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = image.squeeze(-1)
    
    # 确保掩码数据格式正确
    if len(gt_mask.shape) == 3 and gt_mask.shape[0] == 1:
        gt_mask = gt_mask.squeeze(0)
    elif len(gt_mask.shape) == 3 and gt_mask.shape[2] == 1:
        gt_mask = gt_mask.squeeze(-1)
    
    if len(pred_mask.shape) == 3 and pred_mask.shape[0] == 1:
        pred_mask = pred_mask.squeeze(0)
    elif len(pred_mask.shape) == 3 and pred_mask.shape[2] == 1:
        pred_mask = pred_mask.squeeze(-1)
    
    # 原始图像
    plt.subplot(1, 4, 1)
    if len(image.shape) == 3:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # 标注点
    plt.subplot(1, 4, 2)
    if len(image.shape) == 3:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')
    plt.plot(input_point[0][0], input_point[0][1], 'r+', markersize=10, linewidth=2)
    plt.title('Input Point')
    plt.axis('off')
    
    # 真实标注
    plt.subplot(1, 4, 3)
    plt.imshow(gt_mask, cmap='gray')
    plt.title(f'Ground Truth')
    plt.axis('off')
    
    # 预测结果
    plt.subplot(1, 4, 4)
    plt.imshow(pred_mask, cmap='gray')
    plt.title(f'Prediction\nIoU: {iou:.3f}, Dice: {dice:.3f}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'sample_{sample_idx:04d}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 单独保存预测结果图像，调整到原始标签尺寸
    pred_img = (pred_mask * 255).astype(np.uint8)
    
    # 如果提供了原始标签路径，读取原始尺寸并调整预测结果
    if original_mask_path and os.path.exists(original_mask_path):
        try:
            original_mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)
            if original_mask is not None:
                # 调整预测结果到原始标签尺寸
                pred_img = cv2.resize(pred_img, (original_mask.shape[1], original_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            print(f"警告: 无法读取原始标签 {original_mask_path}: {e}")
    
    # 保存到predictions目录，使用与原始标签相同的文件名
    base_dir = os.path.dirname(save_path)
    predictions_dir = os.path.join(base_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    if mask_name:
        pred_filename = mask_name
    else:
        pred_filename = f"prediction_{sample_idx:04d}.png"
    
    cv2.imwrite(os.path.join(predictions_dir, pred_filename), pred_img)

def batch_test(predictor, test_dataset, device, args):
    """批量测试"""
    print("开始批量测试...")
    
    # 创建保存目录
    if args.visualize:
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(os.path.join(args.save_path, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(args.save_path, 'predictions'), exist_ok=True)
    
    # 测试所有样本
    num_samples = len(test_dataset)
    
    results = []
    total_iou = 0
    total_dice = 0
    valid_samples = 0
    
    progress_bar = tqdm(range(num_samples), desc='Testing', unit='sample')
    
    for i in progress_bar:
        try:
            # 读取测试样本
            image, mask, input_point, original_mask_path, mask_name = read_single_for_test(test_dataset, i)
            
            # 测试单张图像
            save_path = os.path.join(args.save_path, 'visualizations') if args.visualize else None
            result = test_single_image(predictor, image, mask, input_point, device, save_path, i, original_mask_path, mask_name)
            
            if result is not None:
                results.append(result)
                total_iou += result['iou']
                total_dice += result['dice']
                valid_samples += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'IoU': f'{result["iou"]:.3f}',
                    'Dice': f'{result["dice"]:.3f}',
                    'Avg_IoU': f'{total_iou/valid_samples:.3f}',
                    'Avg_Dice': f'{total_dice/valid_samples:.3f}'
                })
                    
        except Exception as e:
            print(f"处理样本 {i} 时出错: {str(e)}")
            continue
    
    # 计算平均指标
    if valid_samples > 0:
        avg_iou = total_iou / valid_samples
        avg_dice = total_dice / valid_samples
    else:
        avg_iou = 0
        avg_dice = 0
    
    return results, avg_iou, avg_dice, valid_samples

def save_test_results(results, avg_iou, avg_dice, valid_samples, args):
    """保存测试结果到CSV文件"""
    # 保存为CSV文件
    csv_file = os.path.join(args.save_path, 'test_results.csv')
    with open(csv_file, 'w') as f:
        f.write('sample_idx,iou,dice,prediction_image\n')
        for i, result in enumerate(results):
            # 获取对应的原始标签文件名
            if hasattr(result, 'mask_name') and result['mask_name']:
                pred_image_path = f'predictions/{result["mask_name"]}'
            else:
                pred_image_path = f'predictions/prediction_{i:04d}.png'
            f.write(f'{i},{result["iou"]:.6f},{result["dice"]:.6f},{pred_image_path}\n')
    
    print(f"\n测试结果已保存到: {args.save_path}")
    print(f"CSV结果: {csv_file}")

def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载测试数据集
    print("正在加载测试数据集...")
    test_dataset = FullDataset(args.test_image_path, args.test_mask_path, 352, mode='val')
    print(f"测试数据集大小: {len(test_dataset)}")
    
    # 加载模型
    print("正在加载SAM1模型...")
    try:
        # 构建SAM1模型
        sam = sam_model_registry[args.model_type](checkpoint=None)  # 不使用预训练权重
        
        # 加载训练好的权重
        print(f"正在加载训练好的权重: {args.model_path}")
        state_dict = torch.load(args.model_path, map_location=device)
        sam.load_state_dict(state_dict)
        sam.to(device)
        
        # 创建SamPredictor
        predictor = SamPredictor(sam)
        print("模型加载成功")
        
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        raise
    
    # 设置为评估模式
    sam.eval()
    
    # 打印模型信息
    total_params = sum(p.numel() for p in sam.parameters())
    print(f"模型总参数量: {total_params:,}")
    
    # 批量测试
    results, avg_iou, avg_dice, valid_samples = batch_test(predictor, test_dataset, device, args)
    
    # 打印测试结果
    print(f"\n{'='*50}")
    print("测试结果总结")
    print(f"{'='*50}")
    print(f"总样本数: {len(results)}")
    print(f"有效样本数: {valid_samples}")
    print(f"平均IoU: {avg_iou:.4f}")
    print(f"平均Dice: {avg_dice:.4f}")
    print(f"{'='*50}")
    
    # 保存测试结果
    save_test_results(results, avg_iou, avg_dice, valid_samples, args)

if __name__ == "__main__":
    args = parse_args()
    main(args) 