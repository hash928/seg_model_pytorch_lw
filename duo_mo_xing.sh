# 2024-06-17 17:00:00
# 多模型训练脚本，包含不同模型、损失函数和注意力机制的组合，以评估其在图像分割任务中的性能。


# 论文中结果使用的是checkpoints2作为多模型比较，采样dice损失
# checkpoints5作为注意力机制消融，采样bce_dice损失，但是单独是aspp是使用3368随机数种子
# checkpoints6作为损失函数消融

# 多模型比较
# python train_unet.py \
#     --model_type "unet_resnet18" \
#     --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Images/" \
#     --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Labels/" \
#     --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Images/" \
#     --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Labels/" \
#     --save_path "./checkpoints7/unet_resnet18_dice_loss_training_Mosaic_4_seed33" \
#     --epoch "200" \
#     --lr "1e-4" \
#     --batch_size "8" \
#     --weight_decay "1e-4" \
#     --num_workers "4" \
#     --input_size "352" \
#     --num_classes "1" \
#     --seed "33" \
#     --loss_type "dice"\

# python train_unet.py \
#     --model_type "segnet" \
#     --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Images/" \
#     --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Labels/" \
#     --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Images/" \
#     --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Labels/" \
#     --save_path "./checkpoints7/segnet_dice_loss_training_Mosaic_4_seed33" \
#     --epoch "200" \
#     --lr "1e-4" \
#     --batch_size "8" \
#     --weight_decay "1e-4" \
#     --num_workers "4" \
#     --input_size "352" \
#     --num_classes "1" \
#     --seed "33" \
#     --loss_type "dice"\

# python train_unet.py \
#     --model_type "deconvnet" \
#     --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Images/" \
#     --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Labels/" \
#     --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Images/" \
#     --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Labels/" \
#     --save_path "./checkpoints7/deconvnet_dice_loss_training_Mosaic_4_seed33" \
#     --epoch "200" \
#     --lr "1e-4" \
#     --batch_size "8" \
#     --weight_decay "1e-4" \
#     --num_workers "4" \
#     --input_size "352" \
#     --num_classes "1" \
#     --seed "33" \
#     --loss_type "dice"\




# python train_unet.py \
#     --model_type "pspnet_resnet50" \
#     --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Images/" \
#     --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Labels/" \
#     --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Images/" \
#     --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Labels/" \
#     --save_path "./checkpoints7/pspnet_resnet50_dice_loss_training_Mosaic_4_seed33" \
#     --epoch "200" \
#     --lr "1e-4" \
#     --batch_size "8" \
#     --weight_decay "1e-4" \
#     --num_workers "4" \
#     --input_size "352" \
#     --num_classes "1" \
#     --seed "33" \
#     --loss_type "dice"\

# python train_unet.py \
#     --model_type "unet_base" \
#     --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Images/" \
#     --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Labels/" \
#     --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Images/" \
#     --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Labels/" \
#     --save_path "./checkpoints7/unet_base_dice_loss_training_Mosaic_4_seed33" \
#     --epoch "200" \
#     --lr "1e-4" \
#     --batch_size "8" \
#     --weight_decay "1e-4" \
#     --num_workers "4" \
#     --input_size "352" \
#     --num_classes "1" \
#     --seed "33" \
#     --loss_type "dice"\

#     python train_unet.py \
#     --model_type "deeplabv3p_xception" \
#     --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Images/" \
#     --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Labels/" \
#     --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Images/" \
#     --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Labels/" \
#     --save_path "./checkpoints7/deeplabv3p_xception_dice_loss_training_Mosaic_4_seed33" \
#     --epoch "200" \
#     --lr "1e-4" \
#     --batch_size "8" \
#     --weight_decay "1e-4" \
#     --num_workers "4" \
#     --input_size "352" \
#     --num_classes "1" \
#     --seed "33" \
#     --loss_type "dice"\
#     --xception_width_mult "0.5" \


# 注意力机制消融
# python train_unet.py \
#     --model_type "unet_resnet34" \
#     --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Images/" \
#     --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Labels/" \
#     --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Images/" \
#     --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Labels/" \
#     --save_path "./checkpoints5/unet_resnet34_bce_dice_loss_aspp_se_training_Mosaic_4_seed33" \
#     --epoch "200" \
#     --lr "1e-4" \
#     --batch_size "8" \
#     --weight_decay "1e-4" \
#     --num_workers "4" \
#     --input_size "352" \
#     --num_classes "1" \
#     --seed "33" \
#     --loss_type "bce_dice"\
#     --use_aspp \
#     --use_se \

# python train_unet.py \
#     --model_type "unet_resnet34" \
#     --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Images/" \
#     --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Labels/" \
#     --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Images/" \
#     --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Labels/" \
#     --save_path "./checkpoints5/unet_resnet34_bce_dice_loss_aspp_training_Mosaic_4_seed3368" \
#     --epoch "200" \
#     --lr "1e-4" \
#     --batch_size "8" \
#     --weight_decay "1e-4" \
#     --num_workers "4" \
#     --input_size "352" \
#     --num_classes "1" \
#     --seed "3368" \
#     --loss_type "bce_dice"\
#     --use_aspp \


# python train_unet.py \
#     --model_type "unet_resnet34" \
#     --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Images/" \
#     --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Labels/" \
#     --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Images/" \
#     --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Labels/" \
#     --save_path "./checkpoints5/unet_resnet34_bce_dice_loss_aspp_ca_training_Mosaic_4_seed33" \
#     --epoch "200" \
#     --lr "1e-4" \
#     --batch_size "8" \
#     --weight_decay "1e-4" \
#     --num_workers "4" \
#     --input_size "352" \
#     --num_classes "1" \
#     --seed "33" \
#     --loss_type "bce_dice"\
#     --use_aspp \
#     --use_ca \

# python train_unet.py \
#     --model_type "unet_resnet34" \
#     --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Images/" \
#     --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Labels/" \
#     --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Images/" \
#     --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Labels/" \
#     --save_path "./checkpoints5/unet_resnet34_bce_dice_loss_aspp_cbam_training_Mosaic_4_seed33" \
#     --epoch "200" \
#     --lr "1e-4" \
#     --batch_size "8" \
#     --weight_decay "1e-4" \
#     --num_workers "4" \
#     --input_size "352" \
#     --num_classes "1" \
#     --seed "33" \
#     --loss_type "bce_dice"\
#     --use_aspp \
#     --use_cbam \

# python train_unet.py \
#     --model_type "unet_resnet34" \
#     --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Images/" \
#     --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Labels/" \
#     --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Images/" \
#     --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Labels/" \
#     --save_path "./checkpoints5/unet_resnet34_bce_dice_loss_aspp_eca_training_Mosaic_4_seed33" \
#     --epoch "200" \
#     --lr "1e-4" \
#     --batch_size "8" \
#     --weight_decay "1e-4" \
#     --num_workers "4" \
#     --input_size "352" \
#     --num_classes "1" \
#     --seed "33" \
#     --loss_type "bce_dice"\
#     --use_aspp \
#     --use_eca \


# 损失函数消融
# python train_unet.py \
#     --model_type "unet_resnet34" \
#     --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Images/" \
#     --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Labels/" \
#     --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Images/" \
#     --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Labels/" \
#     --save_path "./checkpoints6/unet_resnet34_bce_loss_aspp_se_training_Mosaic_4_seed33" \
#     --epoch "200" \
#     --lr "1e-4" \
#     --batch_size "8" \
#     --weight_decay "1e-4" \
#     --num_workers "4" \
#     --input_size "352" \
#     --num_classes "1" \
#     --seed "33" \
#     --loss_type "bce"\
#     --use_aspp \
#     --use_se \

# python train_unet.py \
#     --model_type "unet_resnet34" \
#     --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Images/" \
#     --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Labels/" \
#     --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Images/" \
#     --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Labels/" \
#     --save_path "./checkpoints6/unet_resnet34_dice_loss_aspp_se_training_Mosaic_4_seed33" \
#     --epoch "200" \
#     --lr "1e-4" \
#     --batch_size "8" \
#     --weight_decay "1e-4" \
#     --num_workers "4" \
#     --input_size "352" \
#     --num_classes "1" \
#     --seed "33" \
#     --loss_type "dice"\
#     --use_aspp \
#     --use_se \

# python train_unet.py \
#     --model_type "unet_resnet34" \
#     --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Images/" \
#     --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Labels/" \
#     --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Images/" \
#     --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Labels/" \
#     --save_path "./checkpoints6/unet_resnet34_focal_loss_aspp_se_training_Mosaic_4_seed33" \
#     --epoch "200" \
#     --lr "1e-4" \
#     --batch_size "8" \
#     --weight_decay "1e-4" \
#     --num_workers "4" \
#     --input_size "352" \
#     --num_classes "1" \
#     --seed "33" \
#     --loss_type "focal"\
#     --use_aspp \
#     --use_se \

# 组合损失权重消融（w_bce + w_dice = 1；与 checkpoints6 损失消融同配置，仅改 bce_dice 权重）
# G1: 0.25 / 0.75 偏 Dice
python train_unet.py \
    --model_type "unet_resnet34" \
    --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Images/" \
    --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Labels/" \
    --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Images/" \
    --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Labels/" \
    --save_path "./checkpoints8/unet_resnet34_bce_dice_loss_aspp_se_training_Mosaic_4_seed33_bce_dice_wbce0p25_wdice0p75" \
    --epoch "200" \
    --lr "1e-4" \
    --batch_size "8" \
    --weight_decay "1e-4" \
    --num_workers "4" \
    --input_size "352" \
    --num_classes "1" \
    --seed "33" \
    --loss_type "bce_dice"\
    --bce_dice_bce_weight "0.25" \
    --bce_dice_dice_weight "0.75" \
    --use_aspp \
    --use_se \

# G2: 0.50 / 0.50 基线
python train_unet.py \
    --model_type "unet_resnet34" \
    --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Images/" \
    --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Labels/" \
    --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Images/" \
    --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Labels/" \
    --save_path "./checkpoints8/unet_resnet34_bce_dice_loss_aspp_se_training_Mosaic_4_seed33_bce_dice_wbce0p5_wdice0p5" \
    --epoch "200" \
    --lr "1e-4" \
    --batch_size "8" \
    --weight_decay "1e-4" \
    --num_workers "4" \
    --input_size "352" \
    --num_classes "1" \
    --seed "33" \
    --loss_type "bce_dice"\
    --bce_dice_bce_weight "0.5" \
    --bce_dice_dice_weight "0.5" \
    --use_aspp \
    --use_se \

# G3: 0.75 / 0.25 偏 BCE
python train_unet.py \
    --model_type "unet_resnet34" \
    --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Images/" \
    --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Labels/" \
    --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Images/" \
    --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Labels/" \
    --save_path "./checkpoints8/unet_resnet34_bce_dice_loss_aspp_se_training_Mosaic_4_seed33_bce_dice_wbce0p75_wdice0p25" \
    --epoch "200" \
    --lr "1e-4" \
    --batch_size "8" \
    --weight_decay "1e-4" \
    --num_workers "4" \
    --input_size "352" \
    --num_classes "1" \
    --seed "33" \
    --loss_type "bce_dice"\
    --bce_dice_bce_weight "0.75" \
    --bce_dice_dice_weight "0.25" \
    --use_aspp \
    --use_se \

# G4: 0.90 / 0.10 极端偏 BCE
python train_unet.py \
    --model_type "unet_resnet34" \
    --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Images/" \
    --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Labels/" \
    --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Images/" \
    --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Labels/" \
    --save_path "./checkpoints8/unet_resnet34_bce_dice_loss_aspp_se_training_Mosaic_4_seed33_bce_dice_wbce0p9_wdice0p1" \
    --epoch "200" \
    --lr "1e-4" \
    --batch_size "8" \
    --weight_decay "1e-4" \
    --num_workers "4" \
    --input_size "352" \
    --num_classes "1" \
    --seed "33" \
    --loss_type "bce_dice"\
    --bce_dice_bce_weight "0.9" \
    --bce_dice_dice_weight "0.1" \
    --use_aspp \
    --use_se \

# G5: 0.10 / 0.90 极端偏 Dice（与 G4 对称）
python train_unet.py \
    --model_type "unet_resnet34" \
    --train_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Images/" \
    --train_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/aug_Training_Labels/" \
    --val_image_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Images/" \
    --val_mask_path "/home/data/sam-unet/shi_ce/xiangdao_data14/Validation_Labels/" \
    --save_path "./checkpoints8/unet_resnet34_bce_dice_loss_aspp_se_training_Mosaic_4_seed33_bce_dice_wbce0p1_wdice0p9" \
    --epoch "200" \
    --lr "1e-4" \
    --batch_size "8" \
    --weight_decay "1e-4" \
    --num_workers "4" \
    --input_size "352" \
    --num_classes "1" \
    --seed "33" \
    --loss_type "bce_dice"\
    --bce_dice_bce_weight "0.1" \
    --bce_dice_dice_weight "0.9" \
    --use_aspp \
    --use_se \
