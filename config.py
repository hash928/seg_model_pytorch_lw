#!/usr/bin/env python3
# 配置管理模块

import argparse
import re
import sys

def parse_shell_args(shell_file='train_unet.sh'):
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
            
        # 处理布尔参数 - 检查非注释行
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                if '--freeze_backbone' in line:
                    args_dict['freeze_backbone'] = True
                if '--use_aspp' in line:
                    args_dict['use_aspp'] = True
                if '--use_se' in line:
                    args_dict['use_se'] = True
                if '--pretrained' in line:
                    args_dict['pretrained'] = True
            
        print("从shell脚本读取的参数：")
        for key, value in args_dict.items():
            print(f"{key}: {value}")
            
    except FileNotFoundError:
        print(f"警告：未找到shell脚本 {shell_file}，将使用命令行参数")
    except Exception as e:
        print(f"警告：解析shell脚本时出错：{e}，将使用命令行参数")
        
    return args_dict

def create_parser():
    """创建参数解析器"""
    parser = argparse.ArgumentParser("UNet Model Training")
    parser.add_argument("--model_type", type=str, default="unet_base", 
                        choices=["unet_base", "unet_resnet18", "unet_resnet34", "unet_resnet50", "unet_resnet101", "unet_resnet152",
                                "fcn8s", "deeplabv3p_resnet50", "deeplabv3p_resnet101", "deeplabv3p_xception"],
                        help="分割模型类型")
    parser.add_argument("--pretrained", action="store_true", 
                        help="是否使用预训练的backbone（仅对UNet-ResNet模型有效）")
    parser.add_argument("--train_image_path", type=str, required=True, 
                        help="训练图像路径")
    parser.add_argument("--train_mask_path", type=str, required=True,
                        help="训练掩码路径")
    parser.add_argument("--val_image_path", type=str, required=True, 
                        help="验证图像路径")
    parser.add_argument("--val_mask_path", type=str, required=True,
                        help="验证掩码路径")
    parser.add_argument('--save_path', type=str, required=True,
                        help="模型保存路径")
    parser.add_argument("--epoch", type=int, default=200, 
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="是否冻结主干网络参数（支持UNet-ResNet、DeepLabV3+、FCN8s）")
    parser.add_argument("--use_aspp", action="store_true",
                        help="是否使用ASPP模块（仅支持UNet系列模型）")
    parser.add_argument("--use_se", action="store_true",
                        help="是否使用SE模块（仅支持UNet系列模型）")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载时使用的子进程数量")
    parser.add_argument("--input_size", type=int, default=352,
                        help="输入图像尺寸")
    parser.add_argument("--num_classes", type=int, default=1,
                        help="分割类别数（1为二值分割）")
    
    return parser

def parse_args():
    """解析命令行参数，优先从shell脚本读取"""
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
    parser = create_parser()
    args = parser.parse_args()
    
    return args
