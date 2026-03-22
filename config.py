#!/usr/bin/env python3
# 配置管理模块

import argparse
import os
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
                if '--use_cbam' in line:
                    args_dict['use_cbam'] = True
                if '--use_ca' in line:
                    args_dict['use_ca'] = True
                if '--use_eca' in line:
                    args_dict['use_eca'] = True
                if '--attention_type' in line:
                    # 提取attention_type的值
                    match = re.search(r'--attention_type\s+"?(\w+)"?', line)
                    if match:
                        args_dict['attention_type'] = match.group(1)
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
    parser.add_argument(
        "--model_type",
        type=str,
        default="unet_base",
        choices=[
            "unet_base",
            "unet_resnet18",
            "unet_resnet34",
            "unet_resnet50",
            "unet_resnet101",
            "unet_resnet152",
            "fcn8s",
            "fcn_resnet34",
            "fcn_resnet50",
            "fcn32s_vgg16",
            "segnet",
            "deconvnet",
            "refinenet_resnet50",
            "pspnet_resnet50",
            "deeplabv3p_resnet50",
            "deeplabv3p_resnet101",
            "deeplabv3p_xception",
        ],
        help="分割模型类型",
    )
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
                        help="是否使用SE模块（仅支持UNet系列模型，向后兼容）")
    parser.add_argument("--use_cbam", action="store_true",
                        help="是否使用CBAM模块（仅支持UNet系列模型）")
    parser.add_argument("--use_ca", action="store_true",
                        help="是否使用CA模块（仅支持UNet系列模型）")
    parser.add_argument("--use_eca", action="store_true",
                        help="是否使用ECA模块（仅支持UNet系列模型）")
    parser.add_argument("--attention_type", type=str, default=None,
                        choices=["se", "cbam", "ca", "eca"],
                        help="注意力模块类型（优先级高于--use_*参数）")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载时使用的子进程数量")
    parser.add_argument("--input_size", type=int, default=352,
                        help="输入图像尺寸")
    parser.add_argument("--num_classes", type=int, default=1,
                        help="分割类别数（1为二值分割）")

    # 复现 / 性能相关
    parser.add_argument("--seed", type=int, default=33,
                        help="随机数种子（影响数据shuffle、数据增强等）")
    parser.add_argument("--deterministic", action="store_true",
                        help="启用确定性训练（更可复现但通常更慢）")

    # DeepLabV3+ Xception 相关
    parser.add_argument("--xception_width_mult", type=float, default=1.0,
                        help="Xception 宽度系数，<1 会减小通道数以加速/省显存（仅对 deeplabv3p_xception 生效）")
    parser.add_argument("--xception_output_stride", type=int, default=16, choices=[8, 16, 32],
                        help="Xception output_stride（仅对 deeplabv3p_xception 生效；16 通常精度/速度较平衡）")
    
    return parser

def parse_args():
    """解析命令行参数。

    兼容原工作流：直接运行 ``python train_unet.py``（无任何 CLI 参数）时，
    仍从 shell 脚本读取参数（默认 ``train_unet.sh``）。

    若通过其它脚本传入参数（如 ``duo_mo_xing.sh`` 里的多行 ``python train_unet.py --...``），
    只要 ``sys.argv`` 中带有除脚本名以外的参数，则完全以命令行为准，不再覆盖 ``sys.argv``。

    可选：设置环境变量 ``UNET_TRAIN_SHELL`` 指定在无 CLI 参数时要解析的脚本路径
    （默认仍为 ``train_unet.sh``）。
    """
    # 仅在没有传入任何 CLI 参数时使用 shell（保持与原先「只靠 train_unet.sh」一致）
    argv_has_cli = len(sys.argv) > 1
    shell_args = {}
    if not argv_has_cli:
        shell_file = os.environ.get("UNET_TRAIN_SHELL", "train_unet.sh")
        shell_args = parse_shell_args(shell_file)

    if shell_args:
        sys.argv = [sys.argv[0]]  # 清空现有参数
        for key, value in shell_args.items():
            if isinstance(value, bool):
                if value:
                    sys.argv.append(f"--{key}")
            else:
                sys.argv.extend([f"--{key}", str(value)])

    parser = create_parser()
    args = parser.parse_args()
    return args
