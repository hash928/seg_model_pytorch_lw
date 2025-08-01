#!/bin/bash

# SAM1 模型测试脚本
# 使用示例：bash test_sam1.sh

python test_sam1.py \
    --model_path "checkpoints/sam1_training/sam1-best.pth" \
    --model_type "vit_b" \
    --test_image_path "/home/data/sam-unet/xiangdao_data6/Test_Images/" \
    --test_mask_path "/home/data/sam-unet/xiangdao_data6/Test_Labels/" \
    --save_path "checkpoints/sam1_training/test_jie_guo1" \
    --visualize

# 参数说明：
# --model_path: 训练好的SAM1模型路径
# --model_type: SAM1模型类型 (vit_b/vit_l/vit_h)
# --test_image_path: 测试图像路径
# --test_mask_path: 测试标注路径
# --save_path: 测试结果保存路径
# --visualize: 保存可视化结果 