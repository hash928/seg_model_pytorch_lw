# 多模型测试脚本

# 论文中结果使用的是checkpoints2作为多模型比较，采样dice损失
# checkpoints5作为注意力机制消融，采样bce_dice损失，但是单独是aspp是使用3368随机数种子
# checkpoints6作为损失函数消融

# python test_unet.py \
#     --model_path "./checkpoints7/deconvnet_focal_loss_training_Mosaic_4_seed33/deconvnet-best.pth" \
#     --model_type "deconvnet" \
#     --test_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Images/" \
#     --test_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Labels/" \
#     --save_path "./checkpoints7/deconvnet_focal_loss_training_Mosaic_4_seed33/test_results/" \
#     --visualize \
#     --input_size "352" \
#     --num_classes "1" \
#     --batch_size "1" \
#     --threshold "0.5" \


# python test_unet.py \
#     --model_path "./checkpoints7/deeplabv3p_xception_focal_loss_training_Mosaic_4_seed33/deeplabv3p_xception-best.pth" \
#     --model_type "deeplabv3p_xception" \
#     --test_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Images/" \
#     --test_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Labels/" \
#     --save_path "./checkpoints7/deeplabv3p_xception_focal_loss_training_Mosaic_4_seed33/test_results/" \
#     --visualize \
#     --input_size "352" \
#     --num_classes "1" \
#     --batch_size "1" \
#     --threshold "0.5" \
#     --xception_width_mult "0.5" \

# python test_unet.py \
#     --model_path "./checkpoints7/pspnet_resnet50_focal_loss_training_Mosaic_4_seed33/pspnet_resnet50-best.pth" \
#     --model_type "pspnet_resnet50" \
#     --test_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Images/" \
#     --test_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Labels/" \
#     --save_path "./checkpoints7/pspnet_resnet50_focal_loss_training_Mosaic_4_seed33/test_results/" \
#     --visualize \
#     --input_size "352" \
#     --num_classes "1" \
#     --batch_size "1" \
#     --threshold "0.5" \

# python test_unet.py \
#     --model_path "./checkpoints7/segnet_focal_loss_training_Mosaic_4_seed33/segnet-best.pth" \
#     --model_type "segnet" \
#     --test_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Images/" \
#     --test_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Labels/" \
#     --save_path "./checkpoints7/segnet_focal_loss_training_Mosaic_4_seed33/test_results/" \
#     --visualize \
#     --input_size "352" \
#     --num_classes "1" \
#     --batch_size "1" \
#     --threshold "0.5" \

# python test_unet.py \
#     --model_path "./checkpoints7/unet_base_focal_loss_training_Mosaic_4_seed33/unet_base-best.pth" \
#     --model_type "unet_base" \
#     --test_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Images/" \
#     --test_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Labels/" \
#     --save_path "./checkpoints7/unet_base_focal_loss_training_Mosaic_4_seed33/test_results/" \
#     --visualize \
#     --input_size "352" \
#     --num_classes "1" \
#     --batch_size "1" \
#     --threshold "0.5" \


# 注意力机制消融 - 多模型测试脚本
# python test_unet.py \
#     --model_path "./checkpoints5/unet_resnet34_bce_dice_loss_aspp_se_training_Mosaic_4_seed33/unet_resnet34-best.pth" \
#     --model_type "unet_resnet34" \
#     --test_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Images/" \
#     --test_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Labels/" \
#     --save_path "./checkpoints5/unet_resnet34_bce_dice_loss_aspp_se_training_Mosaic_4_seed33/test_results/" \
#     --visualize \
#     --input_size "352" \
#     --num_classes "1" \
#     --batch_size "1" \
#     --threshold "0.5" \
#     --use_aspp \
#     --use_se

python test_unet.py \
    --model_path "./checkpoints5/unet_resnet34_bce_dice_loss_aspp_training_Mosaic_4_seed3368/unet_resnet34-best.pth" \
    --model_type "unet_resnet34" \
    --test_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Images/" \
    --test_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Labels/" \
    --save_path "./checkpoints5/unet_resnet34_bce_dice_loss_aspp_training_Mosaic_4_seed3368/test_results/" \
    --visualize \
    --input_size "352" \
    --num_classes "1" \
    --batch_size "1" \
    --threshold "0.5" \
    --use_aspp

# python test_unet.py \
#     --model_path "./checkpoints5/unet_resnet34_bce_dice_loss_aspp_ca_training_Mosaic_4_seed33/unet_resnet34-best.pth" \
#     --model_type "unet_resnet34" \
#     --test_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Images/" \
#     --test_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Labels/" \
#     --save_path "./checkpoints5/unet_resnet34_bce_dice_loss_aspp_ca_training_Mosaic_4_seed33/test_results/" \
#     --visualize \
#     --input_size "352" \
#     --num_classes "1" \
#     --batch_size "1" \
#     --threshold "0.5" \
#     --use_aspp \
#     --use_ca

# python test_unet.py \
#     --model_path "./checkpoints5/unet_resnet34_bce_dice_loss_aspp_cbam_training_Mosaic_4_seed33/unet_resnet34-best.pth" \
#     --model_type "unet_resnet34" \
#     --test_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Images/" \
#     --test_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Labels/" \
#     --save_path "./checkpoints5/unet_resnet34_bce_dice_loss_aspp_cbam_training_Mosaic_4_seed33/test_results/" \
#     --visualize \
#     --input_size "352" \
#     --num_classes "1" \
#     --batch_size "1" \
#     --threshold "0.5" \
#     --use_aspp \
#     --use_cbam

# python test_unet.py \
#     --model_path "./checkpoints5/unet_resnet34_bce_dice_loss_aspp_eca_training_Mosaic_4_seed33/unet_resnet34-best.pth" \
#     --model_type "unet_resnet34" \
#     --test_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Images/" \
#     --test_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Labels/" \
#     --save_path "./checkpoints5/unet_resnet34_bce_dice_loss_aspp_eca_training_Mosaic_4_seed33/test_results/" \
#     --visualize \
#     --input_size "352" \
#     --num_classes "1" \
#     --batch_size "1" \
#     --threshold "0.5" \
#     --use_aspp \
#     --use_eca

# 损失函数消融
# python test_unet.py \
#     --model_path "./checkpoints6/unet_resnet34_bce_loss_aspp_se_training_Mosaic_4_seed33/unet_resnet34-best.pth" \
#     --model_type "unet_resnet34" \
#     --test_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Images/" \
#     --test_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Labels/" \
#     --save_path "./checkpoints6/unet_resnet34_bce_loss_aspp_se_training_Mosaic_4_seed33/test_results/" \
#     --visualize \
#     --input_size "352" \
#     --num_classes "1" \
#     --batch_size "1" \
#     --threshold "0.5" \
#     --use_aspp \
#     --use_se

# python test_unet.py \
#     --model_path "./checkpoints6/unet_resnet34_dice_loss_aspp_se_training_Mosaic_4_seed33/unet_resnet34-best.pth" \
#     --model_type "unet_resnet34" \
#     --test_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Images/" \
#     --test_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Labels/" \
#     --save_path "./checkpoints6/unet_resnet34_dice_loss_aspp_se_training_Mosaic_4_seed33/test_results/" \
#     --visualize \
#     --input_size "352" \
#     --num_classes "1" \
#     --batch_size "1" \
#     --threshold "0.5" \
#     --use_aspp \
#     --use_se

# python test_unet.py \
#     --model_path "./checkpoints6/unet_resnet34_focal_loss_aspp_se_training_Mosaic_4_seed33/unet_resnet34-best.pth" \
#     --model_type "unet_resnet34" \
#     --test_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Images/" \
#     --test_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Test_Labels/" \
#     --save_path "./checkpoints6/unet_resnet34_focal_loss_aspp_se_training_Mosaic_4_seed33/test_results/" \
#     --visualize \
#     --input_size "352" \
#     --num_classes "1" \
#     --batch_size "1" \
#     --threshold "0.5" \
#     --use_aspp \
#     --use_se