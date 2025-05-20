#!/bin/bash

# 设置使用的GPU设备
export CUDA_VISIBLE_DEVICES="0"

python test.py \
    --checkpoint "/home/seg_model1/save/unetplus/unetplus-best.pth" \
    --test_image_path "/home/data/sam-unet/xiangdao_data3/Test_Images/" \
    --test_mask_path "/home/data/sam-unet/xiangdao_data3/Test_Labels/" \
    --save_path "/home/data/sam-unet/xiangdao_data3/test_jie_guo1/" \
    --model_type "unetplus" \
    --deep_supervision