#!/bin/bash
# UNetb Grad-CAM 可视化示例脚本

# 示例 1: 使用默认设置
python visualize_gradcam.py \
    --image_path "/home/data/sam-unet/shi_ce/xiangdao_data12/Training_Images/yuanxing_jz_label_54.png" \
    --checkpoint "checkpoints/unet_resnet34_aspp_se_dice_bce_loss_training_Mosaic_4_seed512/unet_resnet34-200.pth" \
    --model_type "unet_resnet34" \
    --num_classes 1 \
    --use_aspp \
    --attention_type "se" \
    --layer_name "decoder" \
    --cam_method "xgradcam" \
    --save_path "gradcam_result.png"

# 示例 2: 使用不同的层和方法
python visualize_gradcam.py \
    --image_path "/home/data/sam-unet/shi_ce/xiangdao_data12/Training_Images/yuanxing_jz_label_54.png" \
    --checkpoint "checkpoints/unet_resnet34_aspp_se_dice_bce_loss_training_Mosaic_4_seed512/unet_resnet34-200.pth" \
    --model_type "unet_resnet34" \
    --num_classes 1 \
    --use_aspp \
    --attention_type "se" \
    --layer_name "decoder" \
    --cam_method "gradcam++" \
    --alpha 0.5 \
    --save_path "gradcam_decoder.png"


# 示例 3: 使用 ROI 掩码 最佳参数，后面不要改动这个示例,这个代码每次运行的结果都可能不同，需要多次运行得到最好的，发现是效果与随机数种子有关
python visualize_gradcam.py \
  --image_path "/home/data/sam-unet/shi_ce/xiangdao_data13/Test_Images/buguize2_python_label_219.png" \
  --checkpoint "checkpoints1/unet_resnet34_bce_dice_loss_aspp_se_training_Mosaic_4_seed33/unet_resnet34-best.pth" \
  --model_type "unet_resnet34" \
  --num_classes 1 \
  --use_aspp \
  --attention_type "se" \
  --layer_name "decoder" \
  --cam_method "gradcam" \
  --input_size 352 \
  --use_pred_mask_as_roi \
  --roi_threshold 0.2 \
  --save_path "gradcam_roi.png" \
  --seed 1024
  # --save_mat_path "gradcam_cam.mat"\



python visualize_gradcam.py \
  --image_path "/home/data/sam-unet/shi_ce/xiangdao_data12/aug_Training_Images/jz_label_06_aug00246_01.png" \
  --checkpoint "checkpoints/unet_resnet34_aspp_se_dice_bce_loss_training_Mosaic_4_seed512/unet_resnet34-200.pth" \
  --model_type "unet_resnet34" \
  --num_classes 1 \
  --use_aspp \
  --attention_type "se" \
  --layer_name "decoder" \
  --cam_method "gradcam" \
  --input_size 352 \
  --use_pred_mask_as_roi \
  --roi_mode soft \
  --roi_erode_iters 2 \
  --roi_erode_kernel 5 \
  --save_path "gradcam_roi_inside.png"


