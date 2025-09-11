#!/bin/bash

# UNet 模型测试脚本
# 使用示例：bash test_unet.sh

python test_unet.py \
    --model_path "checkpoints/unet_training/unet-best.pth" \
    --model_type "unet_resnet18" \
    --test_image_path "/home/data/sam-unet/shi_ce/xiangdao_data8/Test_Images/" \
    --test_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data8/Test_Labels/" \
    --save_path "checkpoints/unet_training/test_results/" \
    --visualize \
    --input_size "352" \
    --num_classes "1" \
    --batch_size "1" \
    --threshold "0.5"

# 参数说明：
# --model_path: 训练好的UNet模型路径
# --model_type: UNet模型类型，可选值: "unet_base", "unet_resnet18", "unet_resnet34", "unet_resnet50", "unet_resnet101", "unet_resnet152"
# --test_image_path: 测试图像路径
# --test_mask_path: 测试标注路径
# --save_path: 测试结果保存路径
# --visualize: 保存可视化结果
# --input_size: 输入图像尺寸
# --num_classes: 分割类别数（1为二值分割）
# --batch_size: 测试批次大小
# --threshold: 预测阈值
