import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from mmcv import DataLoader
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import cv2

class ToTensor(object):

    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']
        # 分割 mask 需要保持类别值（0/1 或 0/255），必须使用 NEAREST
        # 否则会把边界插值成灰度“软标签”，导致 dice/IoU 学不到/算不准。
        return {
            'image': F.resize(image, self.size),
            'label': F.resize(label, self.size, interpolation=InterpolationMode.NEAREST),
        }


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}

        return {'image': image, 'label': label}


class RandomBrightness(object):
    def __init__(self, brightness_range=(0.8, 1.2), p=0.5):
        """
        随机调整图像亮度
        brightness_range: 亮度调整范围 (最小值, 最大值)
        p: 应用概率
        """
        self.brightness_range = brightness_range
        self.p = p

    def __call__(self, data):
        if random.random() > self.p:
            return data

        image = data['image']
        label = data['label']

        # 如果图像是PIL格式，转换为numpy数组
        if isinstance(image, Image.Image):
            img_np = np.array(image)
            is_pil = True
        else:
            img_np = image
            is_pil = False

        # 生成随机亮度因子
        brightness_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])

        # 调整亮度
        img_np = np.clip(img_np.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

        # 转换回原始格式
        if is_pil:
            image = Image.fromarray(img_np)
        else:
            image = img_np

        return {'image': image, 'label': label}


class RandomContrast(object):
    def __init__(self, contrast_range=(0.8, 1.2), p=0.5):
        """
        随机调整图像对比度
        contrast_range: 对比度调整范围 (最小值, 最大值)
        p: 应用概率
        """
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, data):
        if random.random() > self.p:
            return data

        image = data['image']
        label = data['label']

        # 如果图像是PIL格式，转换为numpy数组
        if isinstance(image, Image.Image):
            img_np = np.array(image)
            is_pil = True
        else:
            img_np = image
            is_pil = False

        # 生成随机对比度因子
        contrast_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])

        # 计算图像均值作为对比度中心
        mean_val = np.mean(img_np)

        # 调整对比度
        img_np = np.clip((img_np - mean_val) * contrast_factor + mean_val, 0, 255).astype(np.uint8)

        # 转换回原始格式
        if is_pil:
            image = Image.fromarray(img_np)
        else:
            image = img_np

        return {'image': image, 'label': label}


class RandomBlur(object):
    def __init__(self, blur_types=['gaussian', 'median'], kernel_range=(3, 7), p=0.5):
        """
        随机应用模糊处理
        blur_types: 可选的模糊类型 ['gaussian', 'median', 'average']
        kernel_range: 卷积核大小范围 (最小值, 最大值)，必须是奇数
        p: 应用概率
        """
        self.blur_types = blur_types
        self.kernel_range = kernel_range
        self.p = p

    def __call__(self, data):
        if random.random() > self.p:
            return data

        image = data['image']
        label = data['label']

        # 如果图像是PIL格式，转换为numpy数组
        if isinstance(image, Image.Image):
            img_np = np.array(image)
            is_pil = True
        else:
            img_np = image
            is_pil = False

        # 选择随机模糊类型
        blur_type = random.choice(self.blur_types)

        # 随机选择核大小（必须是奇数）
        kernel_size = random.choice(range(self.kernel_range[0], self.kernel_range[1] + 1, 2))

        # 应用不同的模糊方法
        if blur_type == 'gaussian':
            # 高斯模糊
            sigma = random.uniform(0.1, 2.0)  # 随机sigma值
            img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)
        elif blur_type == 'median':
            # 中值模糊
            img_np = cv2.medianBlur(img_np, kernel_size)
        elif blur_type == 'average':
            # 均值模糊
            img_np = cv2.blur(img_np, (kernel_size, kernel_size))

        # 转换回原始格式
        if is_pil:
            image = Image.fromarray(img_np)
        else:
            image = img_np

        return {'image': image, 'label': label}

class RandomRotation(object):
    def __init__(self, degrees=15, p=0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            return {
                'image': image.rotate(angle, Image.BILINEAR, expand=False),
                'label': label.rotate(angle, Image.NEAREST, expand=False)
            }

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, noise_types=['gaussian', 'salt_pepper'], p=0.5):
        """
        Args:
            noise_types: 噪声类型列表，可选 'gaussian', 'salt_pepper'
            p: 应用概率
        """
        self.noise_types = noise_types
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p and isinstance(image, Image.Image):
            img_array = np.array(image).astype(np.float32)
            h, w = img_array.shape[:2]

            # 随机选择一种噪声类型
            noise_type = random.choice(self.noise_types)
            # noise_type = 'salt_pepper'

            if noise_type == 'gaussian':
                # 高斯噪声
                std = random.uniform(0.005, 0.03)  # 随机标准差
                noise = np.random.normal(0, std, img_array.shape)
                img_array = img_array + noise * 255.0

            elif noise_type == 'salt_pepper':
                # 椒盐噪声
                salt_prob = random.uniform(0.001, 0.02)
                pepper_prob = random.uniform(0.001, 0.02)
                random_mask = np.random.random((h, w))

                # 添加盐噪声
                salt_mask = random_mask < salt_prob
                # 添加椒噪声
                pepper_mask = random_mask > (1 - pepper_prob)

                if len(img_array.shape) == 3:
                    img_array[salt_mask] = [255, 255, 255]
                    img_array[pepper_mask] = [0, 0, 0]
                else:
                    img_array[salt_mask] = 255
                    img_array[pepper_mask] = 0

            # 确保值在有效范围内
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

            return {
                'image': Image.fromarray(img_array),
                'label': label
            }

        return {'image': image, 'label': label}


class RandomMosaic(object):
    def __init__(self, p=0.5, dataset=None):
        """
        Mosaic 数据增强：将4张图像拼接成一张
        Args:
            p: 应用概率
            dataset: FullDataset 实例，用于获取其他样本
        """
        self.p = p
        self.dataset = dataset
    
    def set_dataset(self, dataset):
        """设置数据集引用"""
        self.dataset = dataset
    
    def __call__(self, data):
        image, label = data['image'], data['label']
        
        if random.random() < self.p and self.dataset is not None and isinstance(image, Image.Image):
            # 随机选择4个样本的索引
            dataset_size = len(self.dataset)
            if dataset_size < 4:
                return {'image': image, 'label': label}
            
            # 随机选择4个不同的索引
            indices = random.sample(range(dataset_size), 4)
            
            # 加载4张图像和标签
            images = []
            labels = []
            
            for idx in indices[:4]:
                try:
                    img = self.dataset.rgb_loader(self.dataset.images[idx])
                    lbl = self.dataset.binary_loader(self.dataset.gts[idx])
                    images.append(img)
                    labels.append(lbl)
                except:
                    # 如果加载失败，使用当前图像
                    images.append(image.copy())
                    labels.append(label.copy())
            
            # 确保所有图像和标签尺寸一致
            target_size = images[0].size
            images = [img.resize(target_size, Image.BILINEAR) for img in images]
            labels = [lbl.resize(target_size, Image.NEAREST) for lbl in labels]
            
            # 转换为numpy数组进行拼接
            img_arrays = [np.array(img) for img in images]
            lbl_arrays = [np.array(lbl) for lbl in labels]
            
            # 随机选择拼接方式：2x2网格
            # 左上、右上、左下、右下
            h, w = target_size[1], target_size[0]
            half_h, half_w = h // 2, w // 2
            
            # 创建拼接后的图像和标签
            if len(img_arrays[0].shape) == 3:
                mosaic_img = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                mosaic_img = np.zeros((h, w), dtype=np.uint8)
            mosaic_label = np.zeros((h, w), dtype=np.uint8)
            
            # 随机排列4张图像的位置
            positions = [(0, 0), (0, half_w), (half_h, 0), (half_h, half_w)]
            random.shuffle(positions)
            
            for i, (y, x) in enumerate(positions):
                img = img_arrays[i]
                lbl = lbl_arrays[i]
                
                # 裁剪图像到合适大小
                if len(img.shape) == 3:
                    mosaic_img[y:y+half_h, x:x+half_w] = img[:half_h, :half_w]
                else:
                    mosaic_img[y:y+half_h, x:x+half_w] = img[:half_h, :half_w]
                mosaic_label[y:y+half_h, x:x+half_w] = lbl[:half_h, :half_w]
            
            # 转换回PIL Image
            if len(mosaic_img.shape) == 3:
                mosaic_image = Image.fromarray(mosaic_img)
            else:
                mosaic_image = Image.fromarray(mosaic_img).convert('RGB')
            mosaic_label_img = Image.fromarray(mosaic_label).convert('L')
            
            return {'image': mosaic_image, 'label': mosaic_label_img}
        
        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}



class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.size = size
        
        if mode == 'train':
            # 创建 Mosaic transform（需要在 Resize 之前，因为 Mosaic 需要原始尺寸的图像）
            # mosaic_transform = RandomMosaic(p=0.3, dataset=self)
            self.transform = transforms.Compose([
                # mosaic_transform,
                Resize((size, size)),
                RandomNoise(noise_types=['gaussian', 'salt_pepper'], p=0.3),
                RandomHorizontalFlip(p=0.5),
                # RandomVerticalFlip(p=0.5),
                RandomBrightness(brightness_range=(0.8, 1.2), p=0.5),
                RandomContrast(contrast_range=(0.8, 1.2), p=0.5),
                RandomBlur(blur_types=['gaussian', 'median', 'average'], kernel_range=(3, 7), p=0.5),
                RandomRotation(degrees=15),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        try:
            image = self.rgb_loader(self.images[idx])
            label = self.binary_loader(self.gts[idx])
            
            # 确保图像和标签都被调整到正确的尺寸
            data = {'image': image, 'label': label}
            data = self.transform(data)
            
            # 验证输出尺寸
            if data['image'].shape[-2:] != (self.size, self.size):
                print(f"警告：图像 {self.images[idx]} 尺寸不正确: {data['image'].shape}")
                
            if data['label'].shape[-2:] != (self.size, self.size):
                print(f"警告：标签 {self.gts[idx]} 尺寸不正确: {data['label'].shape}")
                
            return data
            
        except Exception as e:
            print(f"加载数据时出错 {self.images[idx]}: {str(e)}")
            # 返回一个默认的空数据
            empty_image = torch.zeros(3, self.size, self.size)
            empty_label = torch.zeros(1, self.size, self.size)
            return {'image': empty_image, 'label': empty_label}

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        

class TestDataset:
    def __init__(self, image_root, gt_root, size):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)

        name = self.images[self.index].split('/')[-1]

        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # 创建训练数据集
    dataset = FullDataset(
        image_root='/home/data/sam-unet/shi_ce/xiangdao_data12/Training_Images/',
        gt_root='/home/data/sam-unet/shi_ce/xiangdao_data12/Training_Labels/',
        size=224,
        mode='train'
    )

    print(f"数据集大小: {len(dataset)}")

    # 选择要可视化的图像索引（可以修改这个值）
    idx = 2017
    print(f"正在可视化第 {idx} 个样本...")

    # 加载原始图像和标签（PIL Image格式）
    original_image = dataset.rgb_loader(dataset.images[idx])
    original_label = dataset.binary_loader(dataset.gts[idx])


    # 反归一化函数
    def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """反归一化图像用于显示"""
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        return tensor * std + mean


    # 逐步应用每个变换，保存每一步的结果
    steps = []
    step_names = []

    # 步骤0: 原始图像
    steps.append({
        'image': original_image.copy(),
        'label': original_label.copy()
    })
    step_names.append('原始图像')

    # 创建初始数据
    current_data = {'image': original_image.copy(), 'label': original_label.copy()}

    # 步骤1: RandomMosaic（可选，如果需要可以取消注释）
    # mosaic_transform = RandomMosaic(p=0.9, dataset=dataset)
    # if hasattr(mosaic_transform, '__call__'):
    #     mosaic_result = mosaic_transform(current_data.copy())
    #     steps.append({
    #         'image': mosaic_result['image'].copy() if hasattr(mosaic_result['image'], 'copy') else mosaic_result['image'],
    #         'label': mosaic_result['label'].copy() if hasattr(mosaic_result['label'], 'copy') else mosaic_result['label']
    #     })
    #     step_names.append('RandomMosaic')
    #     current_data = mosaic_result

    # 步骤2: Resize
    resize_transform = Resize((dataset.size, dataset.size))
    resize_data = resize_transform({'image': original_image.copy(), 'label': original_label.copy()})
    steps.append({
        'image': resize_data['image'].copy() if hasattr(resize_data['image'], 'copy') else resize_data['image'],
        'label': resize_data['label'].copy() if hasattr(resize_data['label'], 'copy') else resize_data['label']
    })
    step_names.append('Resize')

    # 步骤3: RandomNoise
    noise_transform = RandomNoise(noise_types=['gaussian', 'salt_pepper'], p=100)
    noise_data = noise_transform(
        resize_data.copy() if hasattr(resize_data, 'copy') else {'image': resize_data['image'].copy(),
                                                                 'label': resize_data['label'].copy()})
    steps.append({
        'image': noise_data['image'].copy() if hasattr(noise_data['image'], 'copy') else noise_data['image'],
        'label': noise_data['label'].copy() if hasattr(noise_data['label'], 'copy') else noise_data['label']
    })
    step_names.append('RandomNoise')

    # 步骤4: RandomHorizontalFlip (添加这个变换，你之前代码中缺少)
    hflip_transform = RandomHorizontalFlip(p=1)
    hflip_data = hflip_transform({'image': resize_data['image'].copy(), 'label': resize_data['label'].copy()})
    steps.append({
        'image': hflip_data['image'].copy() if hasattr(hflip_data['image'], 'copy') else hflip_data['image'],
        'label': hflip_data['label'].copy() if hasattr(hflip_data['label'], 'copy') else hflip_data['label']
    })
    step_names.append('RandomHorizontalFlip')


    # 步骤6: RandomRotation
    rotation_transform = RandomRotation(degrees=35, p=1)
    rotation_data = rotation_transform({'image': resize_data['image'].copy(), 'label': resize_data['label'].copy()})
    steps.append({
        'image': rotation_data['image'].copy() if hasattr(rotation_data['image'], 'copy') else rotation_data['image'],
        'label': rotation_data['label'].copy() if hasattr(rotation_data['label'], 'copy') else rotation_data['label']
    })
    step_names.append('RandomRotation')

    # 步骤7: RandomColorJitter (可以添加这个变换)
    # color_jitter_transform = RandomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)
    # color_jitter_data = color_jitter_transform({'image': resize_data['image'].copy(), 'label': resize_data['label'].copy()})
    # steps.append({
    #     'image': color_jitter_data['image'].copy() if hasattr(color_jitter_data['image'], 'copy') else color_jitter_data['image'],
    #     'label': color_jitter_data['label'].copy() if hasattr(color_jitter_data['label'], 'copy') else color_jitter_data['label']
    # })
    # step_names.append('RandomColorJitter')

    # 步骤8: RandomBrightness
    Brightness_transform = RandomBrightness(brightness_range=(0.3, 0.3), p=1)
    Brightness_data = Brightness_transform({'image': resize_data['image'].copy(), 'label': resize_data['label'].copy()})
    steps.append({
        'image': Brightness_data['image'].copy() if hasattr(Brightness_data['image'], 'copy') else Brightness_data['image'],
        'label': Brightness_data['label'].copy() if hasattr(Brightness_data['label'], 'copy') else Brightness_data['label']
    })
    step_names.append('RandomBrightness')

    # 步骤9： RandomContrast
    Contrast_transform = RandomContrast(contrast_range=(0.3, 0.3), p=1)
    Contrast_data = Contrast_transform({'image': resize_data['image'].copy(), 'label': resize_data['label'].copy()})
    steps.append({
        'image': Contrast_data['image'].copy() if hasattr(Contrast_data['image'], 'copy') else Contrast_data['image'],
        'label': Contrast_data['label'].copy() if hasattr(Contrast_data['label'], 'copy') else Contrast_data['label']
    })
    step_names.append('RandomContrast')
    # 步骤10 RandomBlur
    Blur_transform = RandomBlur(blur_types=['gaussian', 'median'], kernel_range=(3, 7), p=1)
    Blur_data = Blur_transform({'image': resize_data['image'].copy(), 'label': resize_data['label'].copy()})
    steps.append({
        'image': Blur_data['image'].copy() if hasattr(Blur_data['image'], 'copy') else Blur_data['image'],
        'label': Blur_data['label'].copy() if hasattr(Blur_data['label'], 'copy') else Blur_data['label']
    })
    step_names.append('RandomBlur')

    # 创建可视化图像 - 每个步骤单独保存
    num_steps = len(steps)
    save_dir = 'augmentation_steps'  # 保存目录
    os.makedirs(save_dir, exist_ok=True)  # 创建目录

    for i, (step_data, step_name) in enumerate(zip(steps, step_names)):
        # 创建单独的图形
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # 获取图像和标签数据
        img_data = step_data['image']
        label_data = step_data['label']

        # 处理图像
        if isinstance(img_data, torch.Tensor):
            # 如果是tensor，需要反归一化（如果是normalize后的）
            if i == num_steps - 1:  # Normalize步骤
                img = denormalize(img_data.unsqueeze(0)).squeeze(0)
                img = torch.clamp(img, 0, 1)
            else:
                img = img_data

            # 转换为numpy数组用于显示
            if img.dim() == 3:  # C x H x W
                img_np = img.permute(1, 2, 0).numpy()
            elif img.dim() == 2:  # H x W
                img_np = img.numpy()
                img_np = np.stack([img_np] * 3, axis=-1)  # 转换为3通道
            else:
                img_np = img.numpy()
        else:
            # PIL Image
            img_np = np.array(img_data)
            if len(img_np.shape) == 3:
                # 确保是3通道
                if img_np.shape[2] == 4:  # RGBA转RGB
                    img_np = img_np[:, :, :3]
                img_np = img_np / 255.0
            elif len(img_np.shape) == 2:  # 灰度图转RGB
                img_np = np.stack([img_np] * 3, axis=-1) / 255.0

        # 处理标签
        if isinstance(label_data, torch.Tensor):
            label_np = label_data.squeeze().numpy()
        else:
            label_np = np.array(label_data)

        # 归一化标签到0-1范围（如果最大值大于1）
        if label_np.max() > 1:
            label_np = label_np / 255.0

        # 创建叠加图像
        overlay = img_np.copy()
        mask = label_np > 0.5

        # 确保overlay是3通道
        if len(overlay.shape) == 2:
            overlay = np.stack([overlay] * 3, axis=-1)

        # 应用红色掩码叠加
        overlay[mask] = overlay[mask] * 0.7 + np.array([1, 0, 0]) * 0.3

        # 显示图像
        ax.imshow(overlay)
        ax.set_title(f'{i + 1}. {step_name}', fontsize=12, fontweight='bold')
        ax.axis('off')

        # 添加文件名信息
        filename = os.path.basename(dataset.images[idx])

        # 保存单独图像
        save_path = os.path.join(save_dir, f'step_{i + 1:02d}_{step_name.replace(" ", "_")}.png')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"步骤 {i + 1} 已保存到: {save_path}")

        # 显示图像
        plt.show()

        # 关闭图形以释放内存
        plt.close(fig)

    print("\n数据增强步骤信息:")
    for i, (step_data, step_name) in enumerate(zip(steps, step_names)):
        img_data = step_data['image']
        label_data = step_data['label']

        if isinstance(img_data, torch.Tensor):
            img_shape = img_data.shape
            if i == num_steps - 1:  # Normalize后
                img_range = f"[{img_data.min():.3f}, {img_data.max():.3f}]"
            else:
                img_range = f"[{img_data.min():.3f}, {img_data.max():.3f}]"
        else:
            img_shape = np.array(img_data).shape
            img_range = "[0, 255]"

        if isinstance(label_data, torch.Tensor):
            label_shape = label_data.shape
        else:
            label_shape = np.array(label_data).shape

        print(f"  {i + 1}. {step_name}:")
        print(f"     图像形状: {img_shape}, 值范围: {img_range}")
        print(f"     标签形状: {label_shape}")