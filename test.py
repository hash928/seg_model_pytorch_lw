import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import FullDataset
from models.SAM2UNet import SAM2UNet
from models.UNet import UNet
from models.FCN import FCN
from models.UNetPlusPlus import UNetPlusPlus, UNetPlus
from tqdm import tqdm
import numpy as np
import cv2
import re
import sys

def parse_shell_args(shell_file='test.sh'):
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
            
        print("从shell脚本读取的参数：")
        for key, value in args_dict.items():
            print(f"{key}: {value}")
            
    except FileNotFoundError:
        print(f"警告：未找到shell脚本 {shell_file}，将使用命令行参数")
    except Exception as e:
        print(f"警告：解析shell脚本时出错：{e}，将使用命令行参数")
        
    return args_dict

parser = argparse.ArgumentParser("Model Testing")
parser.add_argument("--model_type", type=str, default="sam2unet", 
                    choices=["sam2unet", "unet", "fcn", "unetplusplus", "unetplus"],
                    help="选择模型类型: sam2unet, unet, fcn, unetplusplus 或 unetplus")
parser.add_argument("--deep_supervision", action="store_true",
                    help="是否使用深度监督（仅当model_type为unetplusplus时有效）")
parser.add_argument("--test_image_path", type=str, required=True, 
                    help="path to the image that used to test the model")
parser.add_argument("--test_mask_path", type=str, required=True,
                    help="path to the mask file for testing")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="path to the model checkpoint")
parser.add_argument("--save_path", type=str, required=True,
                    help="path to save the prediction results")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet34"],
                    help="FCN模型的backbone (仅当model_type为fcn时需要)")

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

def calculate_metrics(pred, mask):
    pred = (pred > 0.5).float()
    intersection = (pred * mask).sum()
    union = (pred + mask).sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred.sum() + mask.sum() + 1e-6)
    return iou.item(), dice.item()

@torch.no_grad()
def test(model, test_loader, device, model_type, save_path, test_mask_path):
    model.eval()
    total_iou = 0
    total_dice = 0
    num_samples = 0
    
    os.makedirs(save_path, exist_ok=True)
    
    # 获取所有测试标签的文件名
    test_masks = sorted([f for f in os.listdir(test_mask_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # 创建指标记录文件
    metrics_file = os.path.join(save_path, 'metrics.csv')
    with open(metrics_file, 'w') as f:
        f.write('Image Name,IoU,Dice\n')
    
    for batch_idx, batch in enumerate(test_loader):
        x = batch['image'].to(device)
        target = batch['label'].to(device)
        
        if model_type == "sam2unet":
            pred0, pred1, pred2 = model(x)
            pred = pred2  # 使用最后一个预测
        elif model_type == "unetplusplus" and isinstance(model, UNetPlusPlus) and model.deep_supervision:
            preds = model(x)
            pred = preds[-1]  # 使用最后一个预测
        else:
            pred = model(x)
            
        pred = torch.sigmoid(pred)
        
        # 保存预测结果
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        
        for i in range(pred_np.shape[0]):
            pred_img = (pred_np[i, 0] * 255).astype(np.uint8)
            # 使用标签图像名称
            mask_name = test_masks[batch_idx * args.batch_size + i]
            
            # 读取原始标签图像以获取尺寸
            mask_path = os.path.join(test_mask_path, mask_name)
            original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if original_mask is None:
                print(f"Warning: Could not read mask {mask_path}")
                continue
                
            # 调整预测结果到原始标签尺寸
            pred_img = cv2.resize(pred_img, (original_mask.shape[1], original_mask.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            # 计算单个图像的指标
            pred_binary = (pred_img > 127).astype(np.float32)
            mask_binary = (original_mask > 127).astype(np.float32)
            
            intersection = np.sum(pred_binary * mask_binary)
            union = np.sum(pred_binary) + np.sum(mask_binary) - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            dice = (2 * intersection + 1e-6) / (np.sum(pred_binary) + np.sum(mask_binary) + 1e-6)
            
            # 保存指标到CSV文件
            with open(metrics_file, 'a') as f:
                f.write(f'{mask_name},{iou:.4f},{dice:.4f}\n')
            
            # 打印单个图像的指标
            print(f'Image: {mask_name}, IoU: {iou:.4f}, Dice: {dice:.4f}')
            
            save_name = os.path.join(save_path, f"pred_{mask_name}")
            cv2.imwrite(save_name, pred_img)
            
            total_iou += iou
            total_dice += dice
            num_samples += 1
    
    # 计算并打印平均指标
    avg_iou = total_iou / num_samples
    avg_dice = total_dice / num_samples
    print(f'\nAverage Metrics:')
    print(f'Average IoU: {avg_iou:.4f}')
    print(f'Average Dice: {avg_dice:.4f}')
    
    # 将平均指标添加到CSV文件
    with open(metrics_file, 'a') as f:
        f.write(f'\nAverage,{avg_iou:.4f},{avg_dice:.4f}\n')
    
    return avg_iou, avg_dice

def main(args):
    # 创建测试数据加载器
    test_dataset = FullDataset(args.test_image_path, args.test_mask_path, 352, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=3)
    
    device = torch.device("cuda")
    
    # 根据模型类型创建模型
    if args.model_type == "sam2unet":
        model = SAM2UNet()
    elif args.model_type == "unet":
        model = UNet(n_channels=3, n_classes=1)
    elif args.model_type == "unetplusplus":
        model = UNetPlusPlus(n_channels=3, n_classes=1, deep_supervision=args.deep_supervision)
    elif args.model_type == "unetplus":
        model = UNetPlus(n_channels=3, n_classes=1)
    else:  # fcn
        model = FCN(n_channels=3, n_classes=1, backbone=args.backbone)
    
    # 使用weights_only=True加载模型权重
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model.to(device)
    
    # 测试模型
    test_iou, test_dice = test(model, test_loader, device, args.model_type, args.save_path, args.test_mask_path)
    print(f'Test IoU: {test_iou:.4f}, Test Dice: {test_dice:.4f}')

if __name__ == "__main__":
    main(args)