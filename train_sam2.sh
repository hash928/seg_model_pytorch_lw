#!/bin/bash

# SAM2 训练脚本
# 使用示例：bash train_sam2.sh

python train_sam2.py \
    --sam2_checkpoint "/home/SAM2-UNet/sam2_hiera_large.pt" \
    --model_cfg "sam2_hiera_l.yaml" \
    --train_image_path "/home/data/sam-unet/xiangdao_data5/Training_Images/" \
    --train_mask_path "/home/data/sam-unet/xiangdao_data5/Training_Labels/" \
    --val_image_path "/home/data/sam-unet/xiangdao_data5/Validation_Images/" \
    --val_mask_path "/home/data/sam-unet/xiangdao_data5/Validation_Labels/" \
    --save_path "./checkpoints/sam2_training" \
    --epoch "200" \
    --lr "1e-5" \
    --batch_size "4" \
    --weight_decay "4e-5" \
    --num_workers "3"

# 可选参数：
# --freeze_backbone  # 冻结主干网络参数 