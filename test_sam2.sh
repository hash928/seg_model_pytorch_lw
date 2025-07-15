#!/bin/bash

# SAM2 模型测试脚本
# 使用示例：bash test_sam2.sh

python test_sam2.py \
    --model_path "./checkpoints/sam2_training/sam2-best.pth" \
    --model_cfg "sam2_configs/sam2_hiera_l.yaml" \
    --test_image_path "/home/data/sam-unet/xiangdao_data5/Test_Images/" \
    --test_mask_path "/home/data/sam-unet/xiangdao_data5/Test_Labels/" \
    --save_path "./test_results" \
    --visualize

# 参数说明：
# --model_path: 训练好的模型路径
# --test_image_path: 测试图像路径
# --test_mask_path: 测试标注路径
# --save_path: 测试结果保存路径
# --visualize: 保存可视化结果 