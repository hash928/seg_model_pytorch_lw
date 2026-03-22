#!/usr/bin/env python3
"""
使用 pytorch-grad-cam 为 UNetb 模型生成 Grad-CAM 热力图可视化
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import cv2

try:
    from scipy.io import savemat
except Exception:
    savemat = None

# 添加项目路径
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.UNetb import unet_resnet, unet_base

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    print("正在安装 pytorch-grad-cam...")
    os.system("pip install grad-cam -q")
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SegmentationModelOutputWrapper(nn.Module):
    """
    将分割模型包装为适合 Grad-CAM 的格式
    对于分割任务，我们选择特定类别的输出作为目标
    """
    def __init__(self, model, target_class=0):
        super().__init__()
        self.model = model
        self.target_class = target_class
    
    def forward(self, x):
        output = self.model(x)
        # 对于分割任务，返回目标类别的输出
        # output shape: [B, C, H, W]
        if output.shape[1] > 1:
            # 多类别分割：选择目标类别
            return output[:, self.target_class:self.target_class+1, :, :]
        else:
            # 二值分割：直接返回
            return output


class SegmentationTarget:
    """
    定义分割模型的 Grad-CAM 目标
    对于分割任务，我们选择特定类别的输出作为目标
    这个类用于指定 Grad-CAM 要可视化的目标
    """
    def __init__(self, target_class=0):
        self.target_class = target_class
    
    def __call__(self, model_output):
        # 对于分割任务，返回目标类别的空间平均激活值
        # model_output shape: [B, C, H, W] 或 [B, 1, H, W]
        # 返回一个标量，用于计算梯度
        if len(model_output.shape) == 4:
            if model_output.shape[1] > 1:
                # 多类别分割：选择目标类别并计算空间平均
                return model_output[:, self.target_class, :, :].sum()
            else:
                # 二值分割：直接计算空间平均
                return model_output[:, 0, :, :].sum()
        else:
            # 如果输出已经是处理过的，直接返回
            return model_output.sum()


class SegmentationRoiTarget:
    def __init__(self, roi_mask: torch.Tensor, target_class: int = 0):
        self.roi_mask = roi_mask
        self.target_class = target_class

    def __call__(self, model_output):
        if len(model_output.shape) == 4:
            if model_output.shape[1] > 1:
                target_map = model_output[:, self.target_class, :, :]
            else:
                target_map = model_output[:, 0, :, :]
        else:
            target_map = model_output

        roi = self.roi_mask
        if roi.dim() == 4:
            roi = roi[:, 0, :, :]
        if roi.dim() == 3:
            roi = roi[0]

        if target_map.dim() == 3:
            target_map = target_map[0]

        roi = roi.to(dtype=target_map.dtype, device=target_map.device)
        if roi.shape != target_map.shape:
            roi = torch.nn.functional.interpolate(
                roi.unsqueeze(0).unsqueeze(0),
                size=target_map.shape[-2:],
                mode='nearest',
            )[0, 0]

        return (target_map * roi).sum()


def get_target_layer(model, layer_name='encoder'):
    """
    获取目标层用于 Grad-CAM
    layer_name 可以是：
    - 'encoder': 编码器最后一层
    - 'decoder': 解码器第一层
    - 'bottleneck': bottleneck 层（如果有 ASPP）
    """
    if layer_name == 'encoder':
        # 获取编码器的最后一层
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'encoder'):
            # 返回最后一个编码器块
            encoder_blocks = model.encoder.encoder
            return encoder_blocks[-4]
        else:
            raise ValueError("无法找到编码器层")
    
    elif layer_name == 'decoder':
        # 获取解码器的第一层（上采样层）
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'ups'):
            return model.decoder.ups[3]
        else:
            raise ValueError("无法找到解码器层")
    
    elif layer_name == 'bottleneck':
        # 获取 bottleneck 层（ASPP 或注意力模块）
        if hasattr(model, 'decoder'):
            if hasattr(model.decoder, 'aspp'):
                return model.decoder.aspp
            elif hasattr(model.decoder, 'attention') and model.decoder.attention is not None:
                return model.decoder.attention
            else:
                # 如果没有 ASPP 或 attention，返回解码器的第一个上采样层
                return model.decoder.ups[0]
        else:
            raise ValueError("无法找到 bottleneck 层")
    
    else:
        raise ValueError(f"未知的层名称: {layer_name}")


def visualize_gradcam(
    model,
    image_path,
    checkpoint_path=None,
    target_class=0,
    layer_name='encoder',
    cam_method='gradcam',
    save_path=None,
    alpha=0.4,
    input_size=None,
    use_pred_mask_as_roi=False,
    roi_threshold=0.5,
    roi_mode='hard',
    roi_erode_iters=0,
    roi_erode_kernel=5,
    save_mat_path=None
):
    """
    使用 Grad-CAM 生成热力图可视化
    
    参数:
    - model: UNet 模型
    - image_path: 输入图像路径
    - checkpoint_path: 模型权重路径（可选）
    - target_class: 目标类别（对于多类别分割）
    - layer_name: 目标层名称 ('encoder', 'decoder', 'bottleneck')
    - cam_method: CAM 方法 ('gradcam', 'gradcam++', 'xgradcam', 'eigencam')
    - save_path: 保存路径（可选）
    - alpha: 热力图透明度 (0-1)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 加载权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"正在加载权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 处理键名映射
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('decoder.se.'):
                new_key = key.replace('decoder.se.', 'decoder.attention.')
                new_state_dict[new_key] = value
            elif 'decoder.classifier' in key:
                print(f"跳过分类器层: {key}")
                continue
            else:
                new_state_dict[key] = value
        
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"警告: 以下键未加载: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"警告: 以下键未使用: {unexpected_keys[:5]}...")
        print("权重加载完成")
    
    # 包装模型
    wrapped_model = SegmentationModelOutputWrapper(model, target_class=target_class)
    
    # 获取目标层
    target_layer = get_target_layer(model, layer_name)
    print(f"使用目标层: {layer_name}")
    
    # 选择 CAM 方法（新版本的 pytorch-grad-cam 会自动检测 CUDA，不需要 use_cuda 参数）
    if cam_method == 'gradcam':
        cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
    elif cam_method == 'gradcam++':
        cam = GradCAMPlusPlus(model=wrapped_model, target_layers=[target_layer])
    elif cam_method == 'xgradcam':
        cam = XGradCAM(model=wrapped_model, target_layers=[target_layer])
    elif cam_method == 'eigencam':
        cam = EigenCAM(model=wrapped_model, target_layers=[target_layer])
    else:
        raise ValueError(f"未知的 CAM 方法: {cam_method}")
    
    # 加载和预处理图像
    print(f"正在加载图像: {image_path}")
    rgb_img = Image.open(image_path).convert("RGB")
    original_size = rgb_img.size
    
    # 转换为 numpy 数组用于可视化
    rgb_img_np = np.array(rgb_img).astype(np.float32) / 255.0
    
    # 预处理图像（转换为 tensor）
    if input_size is not None:
        rgb_img_for_model = rgb_img.resize((input_size, input_size), Image.BILINEAR)
    else:
        rgb_img_for_model = rgb_img

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(rgb_img_for_model).unsqueeze(0).to(device)
    
    # 生成热力图
    print("正在生成 Grad-CAM 热力图...")
    pred_prob_for_mat = None
    roi_mask_for_mat = None
    if use_pred_mask_as_roi:
        with torch.no_grad():
            pred_logits = model(img_tensor)
            if pred_logits.shape[1] > 1:
                pred_prob = torch.softmax(pred_logits, dim=1)[:, target_class:target_class+1]
            else:
                pred_prob = torch.sigmoid(pred_logits)

        pred_prob_for_mat = pred_prob.detach().cpu().numpy()[0, 0]

        if roi_mode == 'soft':
            roi_mask = pred_prob.to(dtype=torch.float32)
        elif roi_mode == 'hard':
            roi_mask = (pred_prob >= roi_threshold).to(dtype=torch.float32)
        else:
            raise ValueError(f"未知的 roi_mode: {roi_mode}")

        if roi_erode_iters and roi_erode_iters > 0:
            roi_np = roi_mask.detach().cpu().numpy()[0, 0]
            roi_bin = (roi_np >= 0.5).astype(np.uint8)
            k = max(1, int(roi_erode_kernel))
            kernel = np.ones((k, k), np.uint8)
            roi_bin = cv2.erode(roi_bin, kernel, iterations=int(roi_erode_iters))
            roi_mask = torch.from_numpy(roi_bin).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)

        roi_mask_for_mat = roi_mask.detach().cpu().numpy()[0, 0]

        targets = [SegmentationRoiTarget(roi_mask=roi_mask, target_class=target_class)]
    else:
        targets = [SegmentationTarget(target_class=target_class)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    if input_size is not None:
        grayscale_cam = cv2.resize(grayscale_cam, original_size, interpolation=cv2.INTER_LINEAR)

    if save_mat_path:
        if savemat is None:
            raise ImportError("scipy 未安装，无法保存 .mat 文件。请先安装 scipy。")
        mat_dict = {
            'grayscale_cam': np.asarray(grayscale_cam, dtype=np.float32),
            'image_path': str(image_path),
            'input_size': -1 if input_size is None else int(input_size),
            'target_class': int(target_class),
            'layer_name': str(layer_name),
            'cam_method': str(cam_method),
            'use_pred_mask_as_roi': bool(use_pred_mask_as_roi),
            'roi_threshold': float(roi_threshold),
            'roi_mode': str(roi_mode),
            'roi_erode_iters': int(roi_erode_iters),
            'roi_erode_kernel': int(roi_erode_kernel),
        }
        if pred_prob_for_mat is not None:
            mat_dict['pred_prob'] = np.asarray(pred_prob_for_mat, dtype=np.float32)
        if roi_mask_for_mat is not None:
            mat_dict['roi_mask'] = np.asarray(roi_mask_for_mat, dtype=np.float32)
        savemat(save_mat_path, mat_dict)
        print(f"CAM 矩阵已保存至: {save_mat_path}")
    
    # 将热力图叠加到原图上
    # show_cam_on_image 会自动处理颜色映射，不需要 colormap 参数
    visualization = show_cam_on_image(
        rgb_img_np,
        grayscale_cam,
        use_rgb=True,
        image_weight=1.0 - alpha
    )
    
    # 创建可视化图像
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(rgb_img)
    axes[0].set_title('原始图像', fontsize=12, fontproperties='SimHei')
    axes[0].axis('off')
    
    # 热力图
    axes[1].imshow(grayscale_cam, cmap='jet')
    axes[1].set_title('Grad-CAM 热力图', fontsize=12, fontproperties='SimHei')
    axes[1].axis('off')
    
    # 叠加图
    axes[2].imshow(visualization)
    axes[2].set_title(f'叠加结果 ({cam_method})', fontsize=12, fontproperties='SimHei')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"结果已保存至: {save_path}")
    else:
        plt.show()
    
    return visualization, grayscale_cam


def main():
    set_seed(1024)
    import argparse
    
    parser = argparse.ArgumentParser(description='UNetb Grad-CAM 可视化')
    parser.add_argument('--image_path', type=str, required=True,
                        help='输入图像路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型权重路径（可选）')
    parser.add_argument('--model_type', type=str, default='unet_resnet34',
                        choices=['unet_base', 'unet_resnet18', 'unet_resnet34', 
                                'unet_resnet50', 'unet_resnet101', 'unet_resnet152'],
                        help='模型类型')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='分割类别数')
    parser.add_argument('--use_aspp', action='store_true',
                        help='是否使用 ASPP 模块')
    parser.add_argument('--attention_type', type=str, default=None,
                        choices=['se', 'cbam', 'ca', 'eca'],
                        help='注意力模块类型')
    parser.add_argument('--target_class', type=int, default=0,
                        help='目标类别（对于多类别分割）')
    parser.add_argument('--layer_name', type=str, default='encoder',
                        choices=['encoder', 'decoder', 'bottleneck'],
                        help='目标层名称')
    parser.add_argument('--cam_method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++', 'xgradcam', 'eigencam'],
                        help='CAM 方法')
    parser.add_argument('--save_path', type=str, default=None,
                        help='保存路径（可选）')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='热力图透明度 (0-1)')

    parser.add_argument('--input_size', type=int, default=None,
                        help='推理输入尺寸（需要与训练一致，例如 224；不填则使用原图尺寸）')
    parser.add_argument('--use_pred_mask_as_roi', action='store_true',
                        help='使用模型预测mask作为ROI，仅在ROI内反传梯度以获得更聚焦的CAM')
    parser.add_argument('--roi_threshold', type=float, default=0.5,
                        help='预测mask二值化阈值')

    parser.add_argument('--roi_mode', type=str, default='hard', choices=['hard', 'soft'],
                        help='ROI 权重模式：hard=二值ROI；soft=使用预测概率图作为权重')
    parser.add_argument('--roi_erode_iters', type=int, default=0,
                        help='对ROI做腐蚀的次数（>0 会让CAM更偏向目标内部）')
    parser.add_argument('--roi_erode_kernel', type=int, default=5,
                        help='ROI腐蚀核大小（奇数更常用，例如 3/5/7）')

    parser.add_argument('--save_mat_path', type=str, default=None,
                        help='保存 .mat 文件路径（可选，包含 grayscale_cam 以及可选 pred_prob/roi_mask）')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='随机数种子')
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # 创建模型
    print(f"正在创建模型: {args.model_type}")
    if args.model_type == 'unet_base':
        model = unet_base(in_channels=3, n_class=args.num_classes,
                         use_aspp=args.use_aspp, attention_type=args.attention_type)
    else:
        resnet_type = args.model_type.replace('unet_', '')
        model = unet_resnet(resnet_type=resnet_type, in_channels=3, n_class=args.num_classes,
                           pretrained=False, use_aspp=args.use_aspp, attention_type=args.attention_type)
    
    # 生成可视化
    visualize_gradcam(
        model=model,
        image_path=args.image_path,
        checkpoint_path=args.checkpoint,
        target_class=args.target_class,
        layer_name=args.layer_name,
        cam_method=args.cam_method,
        save_path=args.save_path,
        alpha=args.alpha,
        input_size=args.input_size,
        use_pred_mask_as_roi=args.use_pred_mask_as_roi,
        roi_threshold=args.roi_threshold,
        roi_mode=args.roi_mode,
        roi_erode_iters=args.roi_erode_iters,
        roi_erode_kernel=args.roi_erode_kernel,
        save_mat_path=args.save_mat_path,
    )


if __name__ == '__main__':
    main()
