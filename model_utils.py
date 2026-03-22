#!/usr/bin/env python3
# 模型工具模块

from models.UNetb import unet_base, unet_resnet
from models.fcn8s import FCN8s
from models.deeplabv3p import DeepLabV3P
from models.FCN import FCN
from models.SegNet import SegNet
from models.FCN_VGG import FCN32s
from models.DeconvNet import DeconvNet
from models.RefineNet import RefineNetResNet50
from models.PSPNet import PSPNetResNet50

def create_model(model_type, num_classes, pretrained=False, use_aspp=False, use_se=False, 
                 use_cbam=False, use_ca=False, use_eca=False, attention_type=None,
                 xception_width_mult: float = 1.0, xception_output_stride: int = 16):
    """创建分割模型
    
    Args:
        model_type: 模型类型
        num_classes: 类别数
        pretrained: 是否使用预训练权重
        use_aspp: 是否使用ASPP模块
        use_se: 是否使用SE模块（向后兼容）
        use_cbam: 是否使用CBAM模块
        use_ca: 是否使用CA模块
        use_eca: 是否使用ECA模块
        attention_type: 注意力模块类型（优先级最高），可选: "se", "cbam", "ca", "eca" 或 None
    """
    # 确定使用的注意力模块类型（优先级：attention_type > use_*参数）
    if attention_type is None:
        if use_se:
            attention_type = "se"
        elif use_cbam:
            attention_type = "cbam"
        elif use_ca:
            attention_type = "ca"
        elif use_eca:
            attention_type = "eca"
        else:
            attention_type = None
    
    if model_type == "unet_base":
        model = unet_base(in_channels=3, n_class=num_classes, use_aspp=use_aspp, attention_type=attention_type)
    elif model_type == "unet_resnet18":
        model = unet_resnet('resnet18', in_channels=3, n_class=num_classes, pretrained=pretrained, use_aspp=use_aspp, attention_type=attention_type)
    elif model_type == "unet_resnet34":
        model = unet_resnet('resnet34', in_channels=3, n_class=num_classes, pretrained=pretrained, use_aspp=use_aspp, attention_type=attention_type)
    elif model_type == "unet_resnet50":
        model = unet_resnet('resnet50', in_channels=3, n_class=num_classes, pretrained=pretrained, use_aspp=use_aspp, attention_type=attention_type)
    elif model_type == "unet_resnet101":
        model = unet_resnet('resnet101', in_channels=3, n_class=num_classes, pretrained=pretrained, use_aspp=use_aspp, attention_type=attention_type)
    elif model_type == "unet_resnet152":
        model = unet_resnet('resnet152', in_channels=3, n_class=num_classes, pretrained=pretrained, use_aspp=use_aspp, attention_type=attention_type)
    elif model_type == "fcn8s":
        # 基于 VGG16 的 FCN8s
        model = FCN8s(n_class=num_classes)
    elif model_type == "fcn32s_vgg16":
        # 原始 VGG16 FCN-32s，不修改主干结构
        model = FCN32s(n_channels=3, n_classes=num_classes, pretrained=pretrained)
    elif model_type == "fcn_resnet34":
        # 基于 ResNet34 的 FCN
        model = FCN(n_channels=3, n_classes=num_classes, backbone='resnet34', pretrained=pretrained)
    elif model_type == "fcn_resnet50":
        # 基于 ResNet50 的 FCN
        model = FCN(n_channels=3, n_classes=num_classes, backbone='resnet50', pretrained=pretrained)
    elif model_type == "deeplabv3p_resnet50":
        model = DeepLabV3P(backbone_type='resnet50', in_channels=3, n_class=num_classes)
    elif model_type == "deeplabv3p_resnet101":
        model = DeepLabV3P(backbone_type='resnet101', in_channels=3, n_class=num_classes)
    elif model_type == "deeplabv3p_xception":
        model = DeepLabV3P(
            backbone_type='xception',
            in_channels=3,
            n_class=num_classes,
            xception_width_mult=xception_width_mult,
            xception_output_stride=xception_output_stride,
        )
    elif model_type == "segnet":
        model = SegNet(n_channels=3, n_classes=num_classes)
    elif model_type == "deconvnet":
        model = DeconvNet(n_channels=3, n_classes=num_classes)
    elif model_type == "refinenet_resnet50":
        model = RefineNetResNet50(n_classes=num_classes, pretrained=pretrained)
    elif model_type == "pspnet_resnet50":
        model = PSPNetResNet50(n_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model

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
    
    print(f"{'='*50}\n")

def freeze_backbone(model, model_type):
    """冻结backbone参数"""
    if 'resnet' in model_type and 'unet' in model_type:
        # UNet + ResNet模型
        for name, param in model.named_parameters():
            if 'encoder' in name and 'layer' in name:
                param.requires_grad = False
        print("已冻结UNet-ResNet backbone参数")
        return True
    elif 'deeplabv3p' in model_type:
        # DeepLabV3+模型
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
        print("已冻结DeepLabV3+ backbone参数")
        return True
    elif model_type == "fcn8s":
        # FCN8s模型（基于VGG16）
        for name, param in model.named_parameters():
            if any(conv_name in name for conv_name in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']):
                param.requires_grad = False
        print("已冻结FCN8s backbone参数")
        return True
    elif model_type.startswith("fcn_resnet"):
        # ResNet 版本的 FCN 模型，冻结 backbone
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
        print("已冻结FCN-ResNet backbone参数")
        return True
    elif model_type == "fcn32s_vgg16":
        # 原始 FCN-32s VGG16 backbone
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
        print("已冻结FCN-32s VGG16 backbone参数")
        return True
    elif model_type == "segnet" or model_type == "deconvnet":
        # SegNet / DeconvNet 结构接近 VGG，可整体训练或不冻结
        print(f"模型类型 {model_type} 默认不单独冻结backbone（整体训练）。")
        return False
    elif model_type in ["refinenet_resnet50", "pspnet_resnet50"]:
        # ResNet50 作为 backbone
        for name, param in model.named_parameters():
            if any(k in name for k in ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]):
                param.requires_grad = False
        print(f"已冻结 {model_type} 中的 ResNet50 backbone 参数")
        return True
    else:
        print(f"模型类型 {model_type} 不支持冻结backbone")
        return False
