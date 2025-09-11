#!/bin/bash

# 设置使用的GPU设备
export CUDA_VISIBLE_DEVICES="0"

# 参数说明：
# --checkpoint: 模型检查点路径
# --test_image_path: 测试图像路径
# --test_mask_path: 测试标签路径
# --save_path: 预测结果保存路径
# --model_type: 模型类型，可选值：
#   - sam2unet: SAM2-UNet模型
#   - sam2: 纯SAM2模型
#   - unet: UNet模型
#   - fcn: FCN模型
#   - unetplusplus: UNet++模型
#   - unetplus: UNet+模型
#   - lstmunet: LSTM-UNet模型
# --hiera_path: SAM2预训练模型路径（仅当model_type为sam2unet或sam2时需要）
# --deep_supervision: 是否使用深度监督（仅当model_type为unetplusplus时有效）
# --backbone: FCN模型的backbone（仅当model_type为fcn时需要），可选值：resnet50, resnet34

# 示例1：测试UNet+模型
python test.py \
    --checkpoint "checkpoints/sam2unet/sam2unet-best.pth" \
    --test_image_path "/home/data/sam-unet/xiangdao_data7/Test_Images/" \
    --test_mask_path "/home/data/sam-unet/xiangdao_data7/Test_Labels/" \
    --save_path "checkpoints/sam2unet/test_jie_guo1/" \
    --model_type "sam2unet" \
    --hiera_path "/home/SAM2-UNet/sam2_hiera_large.pt"
# 示例2：测试SAM2模型
# python test.py \
#     --checkpoint "/home/seg_model1/save/sam2/sam2-best.pth" \
#     --test_image_path "/home/data/sam-unet/xiangdao_data3/Test_Images/" \
#     --test_mask_path "/home/data/sam-unet/xiangdao_data3/Test_Labels/" \
#     --save_path "/home/data/sam-unet/xiangdao_data3/test_jie_guo2/" \
#     --model_type "sam2" \
#     --hiera_path "/path/to/sam2/pretrained/model.pth"

# 示例3：测试LSTM-UNet模型
# python test.py \
#     --checkpoint "/home/seg_model1/save/lstmunet/lstmunet-best.pth" \
#     --test_image_path "/home/data/sam-unet/xiangdao_data3/Test_Images/" \
#     --test_mask_path "/home/data/sam-unet/xiangdao_data3/Test_Labels/" \
#     --save_path "/home/data/sam-unet/xiangdao_data3/test_jie_guo3/" \
#     --model_type "lstmunet"