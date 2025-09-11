#!/bin/bash

# 设置使用的GPU设备
export CUDA_VISIBLE_DEVICES="0"

# 运行训练脚本
python train.py \
    --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data8/Training_Images/" \
    --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data8/Training_Labels/" \
    --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data8/Validation_Images/" \
    --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data8/Validation_Labels/" \
    --save_path "checkpoints/unet/" \
    --epoch "200" \
    --lr "0.0001" \
    --batch_size "4" \
    --model_type "unet" \
    --num_workers "4" \
    # --hiera_path "/home/SAM2-UNet/sam2_hiera_large.pt" \




# 可选参数说明：
# 1. 预训练模型路径
# --pretrained_path "/home/seg_model1/save/sam2unet/sam2unet-best.pth"
# 2. 冻结backbone参数
# --freeze_backbone
# 3. SAM2模型路径
# --hiera_path "/home/SAM2-UNet/sam2_hiera_large.pt"
# --backbone "resnet50"\
# 4. 使用深度监督
# --deep_supervision
# 5. 数据加载时使用的子进程数量
# --num_workers "3"   