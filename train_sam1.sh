#!/bin/bash

python train_sam1.py \
    --sam_checkpoint "/home/seg_model1/pretrained_models/sam_vit_b_01ec64.pth" \
    --model_type "vit_b" \
    --train_image_path "/home/data/sam-unet/xiangdao_data6/Training_Images/" \
    --train_mask_path "/home/data/sam-unet/xiangdao_data6/Training_Labels/" \
    --val_image_path "/home/data/sam-unet/xiangdao_data6/Validation_Images/" \
    --val_mask_path "/home/data/sam-unet/xiangdao_data6/Validation_Labels/" \
    --save_path "./checkpoints/sam1_training" \
    --epoch "20" \
    --lr "1e-5" \
    --weight_decay "4e-5" \
    --batch_size "4" \
    --num_workers "2"
    
# 可选参数:
# --freeze_backbone
# --weight_decay "4e-5" # 权重衰减参数，用于控制L2正则化强度，防止模型过拟合