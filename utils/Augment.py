#!/usr/bin/env python3
# 离线数据增强脚本：支持 Flip / Rotation / Noise / Mosaic，可自由开关
import os
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch

# -------------- 基础工具 --------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def list_image_mask_pairs(image_root, mask_root, exts=(".jpg", ".png", ".jpeg")):
    image_root = Path(image_root)
    mask_root = Path(mask_root)
    images = []
    masks = []
    for img_name in sorted(os.listdir(image_root)):
        if not img_name.lower().endswith(exts):
            continue
        img_path = image_root / img_name
        # 掩码默认同名 .png，如果你实际不一样，自己改这里的规则
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = mask_root / mask_name
        if mask_path.exists():
            images.append(str(img_path))
            masks.append(str(mask_path))
        else:
            print(f"警告：找不到与 {img_path} 对应的掩码 {mask_path}，跳过。")
    return images, masks


def load_rgb(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def load_mask(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("L")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# -------------- 各种增强操作（PIL 版本） --------------

def random_horizontal_flip(img, mask):
    if random.random() < 0.5:
        return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def random_vertical_flip(img, mask):
    if random.random() < 0.5:
        return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask


def random_rotation(img, mask, degrees=15, p=0.5):
    if random.random() >= p:
        return img, mask
    angle = random.uniform(-degrees, degrees)
    img = img.rotate(angle, Image.BILINEAR, expand=False)
    mask = mask.rotate(angle, Image.NEAREST, expand=False)
    return img, mask


def random_noise(img, p=0.1, noise_types=("gaussian", "salt_pepper")):
    if random.random() >= p:
        return img
    if not isinstance(img, Image.Image):
        return img

    img_array = np.array(img).astype(np.float32)
    h, w = img_array.shape[:2]
    noise_type = random.choice(noise_types)

    if noise_type == "gaussian":
        std = random.uniform(0.005, 0.03)
        noise = np.random.normal(0, std, img_array.shape)
        img_array = img_array + noise * 255.0
    elif noise_type == "salt_pepper":
        salt_prob = random.uniform(0.001, 0.02)
        pepper_prob = random.uniform(0.001, 0.02)
        rand_mask = np.random.random((h, w))
        salt_mask = rand_mask < salt_prob
        pepper_mask = rand_mask > (1 - pepper_prob)
        if img_array.ndim == 3:
            img_array[salt_mask] = [255, 255, 255]
            img_array[pepper_mask] = [0, 0, 0]
        else:
            img_array[salt_mask] = 255
            img_array[pepper_mask] = 0

    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def mosaic_4(images, masks, target_size=None):
    """
    简单 2x2 Mosaic，将 4 张 (img, mask) 拼在一起。
    images/masks: list[PIL.Image]，长度至少 4
    target_size: (W, H)，如果为 None，则使用第一张的尺寸
    """
    assert len(images) >= 4 and len(masks) >= 4
    if target_size is None:
        w, h = images[0].size
    else:
        w, h = target_size

    # 统一尺寸
    images = [img.resize((w, h), Image.BILINEAR) for img in images[:4]]
    masks = [m.resize((w, h), Image.NEAREST) for m in masks[:4]]

    half_w, half_h = w // 2, h // 2

    if images[0].mode == "RGB":
        mosaic_img = Image.new("RGB", (w, h))
    else:
        mosaic_img = Image.new("L", (w, h))
    mosaic_mask = Image.new("L", (w, h))

    positions = [(0, 0), (half_w, 0), (0, half_h), (half_w, half_h)]
    random.shuffle(positions)

    for (x, y), img, m in zip(positions, images, masks):
        mosaic_img.paste(img.crop((0, 0, half_w, half_h)), (x, y))
        mosaic_mask.paste(m.crop((0, 0, half_w, half_h)), (x, y))

    return mosaic_img, mosaic_mask


# -------------- 主流程：离线增强 --------------

def augment_offline(
    image_root,
    mask_root,
    out_image_root,
    out_mask_root,
    num_aug_per_image=1,
    use_flip=True,
    use_rotation=True,
    use_noise=True,
    use_mosaic=True,
    mosaic_prob=0.1,
    noise_prob=0.1,
    rotation_degrees=15,
    seed=42,
):
    set_seed(seed)
    ensure_dir(out_image_root)
    ensure_dir(out_mask_root)

    images, masks = list_image_mask_pairs(image_root, mask_root)
    assert len(images) > 0, "未找到任何图像，请检查路径。"

    print(f"共找到 {len(images)} 对 (image, mask)")
    print(f"每张图像将生成 {num_aug_per_image} 个增强样本")
    print(f"增强选项: flip={use_flip}, rotation={use_rotation}, noise={use_noise}, mosaic={use_mosaic}")

    for idx, (img_path, mask_path) in enumerate(zip(images, masks)):
        base_img = load_rgb(img_path)
        base_mask = load_mask(mask_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        for k in range(num_aug_per_image):
            img = base_img.copy()
            msk = base_mask.copy()

            # 先决定是否做 Mosaic（如果启用）
            if use_mosaic and random.random() < mosaic_prob and len(images) >= 4:
                # 随机选另外 3 张图
                other_indices = list(range(len(images)))
                other_indices.remove(idx)
                chosen = random.sample(other_indices, 3)
                imgs = [img] + [load_rgb(images[i]) for i in chosen]
                msks = [msk] + [load_mask(masks[i]) for i in chosen]
                img, msk = mosaic_4(imgs, msks)

            # Flip
            if use_flip:
                img, msk = random_horizontal_flip(img, msk)
                img, msk = random_vertical_flip(img, msk)

            # Rotation
            if use_rotation:
                img, msk = random_rotation(img, msk, degrees=rotation_degrees, p=0.5)

            # Noise
            if use_noise:
                img = random_noise(img, p=noise_prob, noise_types=("gaussian", "salt_pepper"))

            # 保存
            aug_suffix = f"_aug{idx:05d}_{k:02d}"
            out_img_name = img_name + aug_suffix + ".png"
            out_msk_name = img_name + aug_suffix + ".png"

            img.save(os.path.join(out_image_root, out_img_name))
            msk.save(os.path.join(out_mask_root, out_msk_name))

        if (idx + 1) % 50 == 0:
            print(f"已处理 {idx + 1}/{len(images)} 张图像")

    print("离线数据增强完成。")


# -------------- 命令行入口 --------------

def parse_args():
    parser = argparse.ArgumentParser("Offline Data Augmentation")
    parser.add_argument("--image_root", type=str, required=True, help="原始训练图像目录")
    parser.add_argument("--mask_root", type=str, required=True, help="原始训练掩码目录")
    parser.add_argument("--out_image_root", type=str, required=True, help="增强后图像保存目录")
    parser.add_argument("--out_mask_root", type=str, required=True, help="增强后掩码保存目录")
    parser.add_argument("--num_aug_per_image", type=int, default=1, help="每张原始图像生成多少个增强样本")

    # 开关
    parser.add_argument("--no_flip", action="store_true", help="关闭 Flip 增强")
    parser.add_argument("--no_rotation", action="store_true", help="关闭 Rotation 增强")
    parser.add_argument("--no_noise", action="store_true", help="关闭 Noise 增强")
    parser.add_argument("--no_mosaic", action="store_true", help="关闭 Mosaic 增强")

    # 概率与强度
    parser.add_argument("--mosaic_prob", type=float, default=0.1, help="应用 Mosaic 的概率")
    parser.add_argument("--noise_prob", type=float, default=0.1, help="应用 Noise 的概率")
    parser.add_argument("--rotation_degrees", type=float, default=15.0, help="随机旋转角度范围（±deg）")

    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main():
    args = parse_args()
    augment_offline(
        image_root=args.image_root,
        mask_root=args.mask_root,
        out_image_root=args.out_image_root,
        out_mask_root=args.out_mask_root,
        num_aug_per_image=args.num_aug_per_image,
        use_flip=not args.no_flip,
        use_rotation=not args.no_rotation,
        use_noise=not args.no_noise,
        use_mosaic=not args.no_mosaic,
        mosaic_prob=args.mosaic_prob,
        noise_prob=args.noise_prob,
        rotation_degrees=args.rotation_degrees,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()