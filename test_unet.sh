#!/bin/bash

# UNet 模型测试脚本
# 使用示例：bash test_unet.sh

#  python test_unet.py \
#     --model_path "./checkpoints/unet_resnet34_aspp_se_dice_bce_loss_training_Mosaic_4_seed512/unet_resnet34-best.pth" \
#     --model_type "unet_resnet34" \
#     --test_image_path "/home/data/sam-unet/shi_ce/xiangdao_data12/Test_Images/" \
#     --test_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data12/Test_Labels/" \
#     --save_path "./checkpoints/unet_resnet34_aspp_se_dice_bce_loss_training_Mosaic_4_seed512/test_results/" \
#     --visualize \
#     --input_size "352" \
#     --num_classes "1" \
#     --batch_size "1" \
#     --threshold "0.5" \
#     --use_aspp \
#     --use_se \

  python test_unet.py \
     --model_path "./checkpoints1/unet_resnet34_dice_bce_loss_training_seed33/unet_resnet34-200.pth" \
     --model_type "unet_resnet34" \
     --test_image_path "/home/data/sam-unet/shi_ce/data_seg/Test_Images/" \
     --test_mask_path "/home/data/sam-unet/shi_ce/data_seg/Test_Labels/" \
     --save_path "./checkpoints1/unet_resnet34_dice_bce_loss_training_seed33/test_results/" \
     --visualize \
     --input_size "352" \
     --num_classes "1" \
     --batch_size "1" \
     --threshold "0.5" \
#    --use_aspp \
#    --use_se \

# 参数说明：
# --model_path: 训练好的模型路径
# --model_type: 分割模型类型，可选值: 
#   UNet系列: "unet_base", "unet_resnet18", "unet_resnet34", "unet_resnet50", "unet_resnet101", "unet_resnet152"
#   FCN系列: "fcn8s"
#   DeepLabV3+系列: "deeplabv3p_resnet50", "deeplabv3p_resnet101", "deeplabv3p_xception"
# --test_image_path: 测试图像路径
# --test_mask_path: 测试标注路径
# --save_path: 测试结果保存路径
# --visualize: 保存可视化结果
# --input_size: 输入图像尺寸
# --num_classes: 分割类别数（1为二值分割）
# --batch_size: 测试批次大小
# --threshold: 预测阈值

#!/bin/bash

# FCN8s 优化训练脚本

#python test_unet.py \
#    --model_type "fcn8s" \
#     --model_path "checkpoints/fcn8s_training/fcn8s-best.pth" \
#     --model_type "fcn8s" \
#     --test_image_path "/home/data/sam-unet/shi_ce/xiangdao_data8/Test_Images/" \
#     --test_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data8/Test_Labels/" \
#     --save_path "checkpoints/fcn8s_training/test_results/" \
#     --visualize \
#     --input_size "256" \
#     --num_classes "1" \
#     --batch_size "1" \
#     --threshold "0.5"

# 优化说明：
# --batch_size "2"     # 减少批次大小，降低内存需求
# --input_size "256"   # 减少输入尺寸，降低计算量
# --freeze_backbone    # 冻结VGG16 backbone，只训练分类头
# --lr "5e-5"          # 降低学习率，适合冻结backbone的训练


