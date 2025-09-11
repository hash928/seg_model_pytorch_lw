#!/bin/bash

# UNet 训练脚本
# 使用示例：bash train_unet.sh

python train_unet.py \
    --model_type "unet_resnet18" \
    --pretrained \
    --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data8/Training_Images/" \
    --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data8/Training_Labels/" \
    --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data8/Validation_Images/" \
    --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data8/Validation_Labels/" \
    --save_path "./checkpoints/unet_training" \
    --epoch "200" \
    --lr "1e-4" \
    --batch_size "8" \
    --weight_decay "1e-4" \
    --num_workers "4" \
    --input_size "352" \
    --num_classes "1"

# 可选参数：
# --freeze_backbone  # 冻结主干网络参数
# --model_type 可选值: "unet_base", "unet_resnet18", "unet_resnet34", "unet_resnet50", "unet_resnet101", "unet_resnet152"
