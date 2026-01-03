#!/bin/bash

# UNet 训练脚本
# 使用示例：bash train_unet.sh

python train_unet.py \
    --model_type "fcn8s" \
    --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data12/aug_Training_Images/" \
    --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data12/aug_Training_Labels/" \
    --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data12/Validation_Images/" \
    --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data12/Validation_Labels/" \
    --save_path "./checkpoints/fcn8s_dice_loss_training_Mosaic_4_seed33" \
    --epoch "200" \
    --lr "1e-4" \
    --batch_size "8" \
    --weight_decay "1e-4" \
    --num_workers "4" \
    --input_size "352" \
    --num_classes "1" \
#    --use_aspp \
#    --use_eca \

# 可选参数：
# --freeze_backbone  # 冻结主干网络参数
# --model_type 可选值: 
#   UNet系列: "unet_base", "unet_resnet18", "unet_resnet34", "unet_resnet50", "unet_resnet101", "unet_resnet152"
#   FCN系列: "fcn8s"
#   DeepLabV3+系列: "deeplabv3p_resnet50", "deeplabv3p_resnet101", "deeplabv3p_xception"

# 模块说明：
# --use_aspp: 启用ASPP模块，增强多尺度特征提取能力
# --model_type: 支持所有UNet系列模型
# --batch_size "4": ASPP模块会增加内存使用，建议减少批次大小
# ASPP模块可能需要更小的学习率
# --lr "5e-5"

# 注意力模块说明：
# 支持以下注意力模块（仅UNet系列模型）：
# 1. SE模块（Squeeze-and-Excitation）：通道注意力机制
#    --use_se 或 --attention_type "se"
# 2. CBAM模块（Convolutional Block Attention Module）：通道+空间注意力
#    --use_cbam 或 --attention_type "cbam"
# 3. CA模块（Coordinate Attention）：坐标注意力机制
#    --use_ca 或 --attention_type "ca"
# 4. ECA模块（Efficient Channel Attention）：高效通道注意力
#    --use_eca 或 --attention_type "eca"
#
# 可选参数组合：
# 1. 仅使用SE模块：--use_se
# 2. 使用CBAM模块：--use_cbam 或 --attention_type "cbam"
# 3. 使用CA模块：--use_ca 或 --attention_type "ca"
# 4. 使用ECA模块：--use_eca 或 --attention_type "eca"
# 5. SE + ASPP组合：--use_se --use_aspp
# 6. CBAM + ASPP组合：--use_cbam --use_aspp
# 7. 完整组合：--attention_type "cbam" --use_aspp --pretrained
#
# 注意：--attention_type 参数优先级高于 --use_* 参数

#!/bin/bash

# FCN8s 优化训练脚本

#python train_unet.py \
#    --model_type "fcn8s" \
#    --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data8/Training_Images/" \
#    --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data8/Training_Labels/" \
#    --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data8/Validation_Images/" \
#    --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data8/Validation_Labels/" \
#    --save_path "./checkpoints/fcn8s_training" \
#    --epoch "200" \
#    --lr "5e-5" \
#    --batch_size "2" \
#    --weight_decay "1e-4" \
#    --freeze_backbone \
#    --num_workers "4" \
#    --input_size "256" \
#    --num_classes "1"

# 优化说明：
# --batch_size "2"     # 减少批次大小，降低内存需求
# --input_size "256"   # 减少输入尺寸，降低计算量
# --freeze_backbone    # 冻结VGG16 backbone，只训练分类头
# --lr "5e-5"          # 降低学习率，适合冻结backbone的训练
