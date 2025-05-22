#!/bin/bash

# 设置使用的GPU设备
export CUDA_VISIBLE_DEVICES="0"

python test.py \
    --checkpoint "/home/seg_model1/save/sam2unet1/sam2unet-10.pth" \
    --test_image_path "/home/data/sam-unet/xiangdao_duo_data1/Test_Images/" \
    --test_mask_path "/home/data/sam-unet/xiangdao_duo_data1/Test_Labels/" \
    --save_path "/home/data/sam-unet/xiangdao_duo_data1/test_jie_guo1/" \
    --model_type "sam2unet" \
    --class_config "0:249,250,20,道路;1:77,203,129,0,0,0,建筑物;2:61,38,168,背景" \

# 可选参数说明：
# 1. 预训练模型路径
# --checkpoint "/home/seg_model1/save/sam2unet/sam2unet-best.pth"
# 2. 类别配置
# --class_config "0:249,250,20,道路;1:77,203,129,0,0,0,建筑物;2:61,38,168,背景" \
# 3. 模型类型
# --model_type "sam2unet"