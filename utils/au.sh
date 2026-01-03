python Augment.py \
  --image_root "/home/data/sam-unet/shi_ce/xiangdao_data12/Training_Images/" \
  --mask_root "/home/data/sam-unet/shi_ce/xiangdao_data12/Training_Labels/" \
  --out_image_root "/home/data/sam-unet/shi_ce/xiangdao_data12/aug_Training_Images/" \
  --out_mask_root "/home/data/sam-unet/shi_ce/xiangdao_data12/aug_Training_Labels/" \
  --num_aug_per_image 2 \
  --mosaic_prob 0.1 \
  --noise_prob 0.1

python Augment.py \
  --image_root "/home/data/sam-unet/shi_ce/xiangdao_data12/Training_Images/" \
  --mask_root "/home/data/sam-unet/shi_ce/xiangdao_data12/Training_Labels/" \
  --out_image_root "/home/data/sam-unet/shi_ce/xiangdao_data12/aug_Training_Images/" \
  --out_mask_root "/home/data/sam-unet/shi_ce/xiangdao_data12/aug_Training_Labels/" \
  --num_aug_per_image 2 \
  --mosaic_prob 0.1 \
  --no_noise \
  --no_flip \
  --no_rotation
