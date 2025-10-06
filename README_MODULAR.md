# UNet训练代码 - 模块化版本

## 概述

原始的 `train_unet.py` 文件已经重构为模块化设计，提高了代码的可维护性、可读性和可扩展性。

## 模块结构

### 1. 主文件
- **`train_unet.py`** - 主入口文件，现在只有50行代码，负责协调各个模块

### 2. 配置管理
- **`config.py`** - 参数解析和配置管理
  - `parse_shell_args()` - 从shell脚本解析参数
  - `create_parser()` - 创建命令行参数解析器
  - `parse_args()` - 统一参数解析入口

### 3. 模型工具
- **`model_utils.py`** - 模型相关工具函数
  - `create_model()` - 创建UNet模型
  - `print_model_structure()` - 打印模型结构信息
  - `freeze_backbone()` - 冻结backbone参数

### 4. 训练器
- **`trainer.py`** - 训练逻辑封装
  - `UNetTrainer` 类 - 完整的训练器实现
  - 包含训练、验证、模型保存等所有训练相关功能

### 5. 工具函数
- **`utils/`** - 工具函数包
  - **`losses.py`** - 损失函数集合
    - `dice_loss()` - Dice损失
    - `bce_dice_loss()` - BCE+Dice组合损失
    - `focal_loss()` - Focal损失
    - `structure_loss()` - 结构损失
  - **`metrics.py`** - 指标计算
    - `calculate_metrics()` - 计算IoU和Dice指标
    - `calculate_batch_metrics()` - 批次级别指标计算

## 优势

### 1. 代码组织
- **单一职责原则** - 每个模块只负责特定功能
- **低耦合高内聚** - 模块间依赖关系清晰
- **易于维护** - 修改某个功能只需要修改对应模块

### 2. 可扩展性
- **添加新损失函数** - 只需在 `utils/losses.py` 中添加
- **添加新模型** - 只需在 `model_utils.py` 中扩展
- **修改训练逻辑** - 只需修改 `trainer.py`

### 3. 可重用性
- **独立使用** - 各个模块可以独立导入使用
- **测试友好** - 每个模块都可以单独测试
- **配置灵活** - 通过 `config.py` 统一管理配置

## 支持的模型

### UNet系列
- `unet_base` - 基础UNet模型
- `unet_resnet18` - ResNet18 backbone UNet
- `unet_resnet34` - ResNet34 backbone UNet  
- `unet_resnet50` - ResNet50 backbone UNet
- `unet_resnet101` - ResNet101 backbone UNet
- `unet_resnet152` - ResNet152 backbone UNet

### FCN系列
- `fcn8s` - FCN8s模型（基于VGG16 backbone）

### DeepLabV3+系列
- `deeplabv3p_resnet50` - DeepLabV3+ with ResNet50 backbone
- `deeplabv3p_resnet101` - DeepLabV3+ with ResNet101 backbone
- `deeplabv3p_xception` - DeepLabV3+ with Xception backbone

## 使用方法

### 基本使用
```bash
# UNet模型
python train_unet.py --model_type "unet_resnet50" --pretrained \
                     --train_image_path /path/to/train/images \
                     --train_mask_path /path/to/train/masks \
                     --val_image_path /path/to/val/images \
                     --val_mask_path /path/to/val/masks \
                     --save_path /path/to/save/models

# FCN8s模型
python train_unet.py --model_type "fcn8s" \
                     --train_image_path /path/to/train/images \
                     --train_mask_path /path/to/train/masks \
                     --val_image_path /path/to/val/images \
                     --val_mask_path /path/to/val/masks \
                     --save_path /path/to/save/models

# DeepLabV3+模型
python train_unet.py --model_type "deeplabv3p_resnet101" \
                     --train_image_path /path/to/train/images \
                     --train_mask_path /path/to/train/masks \
                     --val_image_path /path/to/val/images \
                     --val_mask_path /path/to/val/masks \
                     --save_path /path/to/save/models
```

### 使用shell脚本
```bash
# UNet训练
bash train_unet.sh

# FCN8s训练
bash train_fcn8s.sh

# DeepLabV3+训练
bash train_deeplabv3p.sh
```

### 编程方式使用
```python
from config import parse_args
from trainer import UNetTrainer
from dataset import FullDataset

# 解析参数
args = parse_args()

# 创建数据集
train_dataset = FullDataset(args.train_image_path, args.train_mask_path, args.input_size, mode='train')
val_dataset = FullDataset(args.val_image_path, args.val_mask_path, args.input_size, mode='val')

# 创建训练器并训练
trainer = UNetTrainer(args, train_dataset, val_dataset)
trainer.train()
```

## 文件对比

| 原始文件 | 模块化后 | 行数减少 |
|---------|---------|---------|
| `train_unet.py` (471行) | `train_unet.py` (51行) | 89% |
| - | `config.py` (85行) | - |
| - | `model_utils.py` (45行) | - |
| - | `trainer.py` (200行) | - |
| - | `utils/losses.py` (50行) | - |
| - | `utils/metrics.py` (20行) | - |

## 模型特点

### FCN8s
- **Backbone**: VGG16
- **特点**: 全卷积网络，使用跳跃连接进行特征融合
- **适用场景**: 语义分割任务，对细节保持较好
- **内存需求**: 中等

### DeepLabV3+
- **Backbone**: ResNet50/101 或 Xception
- **特点**: 使用ASPP模块捕获多尺度特征，结合低级特征
- **适用场景**: 复杂场景的语义分割，对多尺度目标效果好
- **内存需求**: 较高

### UNet系列
- **Backbone**: 自定义或ResNet系列
- **特点**: U型结构，跳跃连接丰富，适合医学图像分割
- **适用场景**: 医学图像、小目标分割
- **内存需求**: 低到中等

## 测试新模型

运行测试脚本验证新模型是否正常工作：
```bash
python test_new_models.py
```

## 扩展建议

1. **添加新的损失函数** - 在 `utils/losses.py` 中添加
2. **添加新的指标** - 在 `utils/metrics.py` 中添加
3. **支持新的模型架构** - 在 `model_utils.py` 中扩展
4. **添加数据增强** - 可以创建 `data_augmentation.py` 模块
5. **添加可视化功能** - 可以创建 `visualization.py` 模块
6. **添加日志系统** - 可以创建 `logger.py` 模块

## 注意事项

1. 确保所有依赖模块都在Python路径中
2. 保持模块间的接口一致性
3. 添加新功能时遵循现有的模块化设计原则
4. 定期重构和优化代码结构
