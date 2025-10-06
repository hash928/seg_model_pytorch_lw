# 分割模型总结

## 🎯 支持的模型列表

### 1. UNet系列
| 模型类型 | 参数量 | 内存需求 | 训练速度 | 特点 |
|---------|--------|---------|---------|------|
| `unet_base` | 31M | 低 | 快 | 基础UNet，适合小目标分割 |
| `unet_resnet18` | ~12M | 低 | 很快 | ResNet18 backbone |
| `unet_resnet34` | ~22M | 低 | 快 | ResNet34 backbone |
| `unet_resnet50` | 72M | 中等 | 中等 | ResNet50 backbone，平衡性能 |
| `unet_resnet101` | ~90M | 中等 | 慢 | ResNet101 backbone |
| `unet_resnet152` | ~110M | 高 | 很慢 | ResNet152 backbone |

### 2. FCN系列
| 模型类型 | 参数量 | 内存需求 | 训练速度 | 特点 |
|---------|--------|---------|---------|------|
| `fcn8s` | 134M | 中等 | 中等 | VGG16 backbone，全卷积网络 |

### 3. DeepLabV3+系列
| 模型类型 | 参数量 | 内存需求 | 训练速度 | 特点 |
|---------|--------|---------|---------|------|
| `deeplabv3p_resnet50` | 67M | 高 | 慢 | ResNet50 + ASPP模块 |
| `deeplabv3p_resnet101` | 86M | 很高 | 很慢 | ResNet101 + ASPP模块 |
| `deeplabv3p_xception` | 30M | 高 | 慢 | Xception + ASPP模块 |

## 🚀 推荐使用场景

### 医学图像分割
- **推荐**: `unet_base`, `unet_resnet50`
- **原因**: UNet的U型结构特别适合医学图像，跳跃连接保持细节

### 一般语义分割
- **推荐**: `fcn8s`, `unet_resnet50`
- **原因**: 平衡性能和精度，适合大多数场景

### 复杂场景分割
- **推荐**: `deeplabv3p_resnet101`, `deeplabv3p_xception`
- **原因**: ASPP模块能捕获多尺度特征，适合复杂场景

### 资源受限环境
- **推荐**: `unet_base`, `unet_resnet18`
- **原因**: 参数量少，内存需求低，训练快速

## 📊 性能对比

### 参数量排序（从小到大）
1. `unet_resnet18` (~12M)
2. `unet_resnet34` (~22M)
3. `deeplabv3p_xception` (30M)
4. `unet_base` (31M)
5. `deeplabv3p_resnet50` (67M)
6. `unet_resnet50` (72M)
7. `deeplabv3p_resnet101` (86M)
8. `unet_resnet101` (~90M)
9. `unet_resnet152` (~110M)
10. `fcn8s` (134M)

### 训练速度排序（从快到慢）
1. `unet_resnet18` - 很快
2. `unet_base` - 快
3. `unet_resnet34` - 快
4. `unet_resnet50` - 中等
5. `fcn8s` - 中等
6. `deeplabv3p_xception` - 慢
7. `deeplabv3p_resnet50` - 慢
8. `unet_resnet101` - 慢
9. `deeplabv3p_resnet101` - 很慢
10. `unet_resnet152` - 很慢

## 🔧 使用建议

### 批次大小建议
- **小模型** (`unet_base`, `unet_resnet18`): batch_size = 8-16
- **中等模型** (`unet_resnet50`, `fcn8s`): batch_size = 4-8
- **大模型** (`deeplabv3p_*`, `unet_resnet152`): batch_size = 2-4

### 学习率建议
- **所有模型**: 1e-4 (默认)
- **预训练模型**: 可以尝试更小的学习率，如 5e-5

### 冻结backbone建议
- **预训练模型**: 可以冻结backbone进行微调
- **从头训练**: 不冻结backbone
- **数据量少**: 建议冻结backbone

## 📝 训练脚本使用

### 快速开始
```bash
# UNet训练
bash train_unet.sh

# FCN8s训练
bash train_fcn8s.sh

# DeepLabV3+训练
bash train_deeplabv3p.sh
```

### 自定义训练
```bash
python train_unet.py \
    --model_type "deeplabv3p_resnet101" \
    --freeze_backbone \
    --batch_size 4 \
    --lr 5e-5 \
    --其他参数...
```

## ✅ 测试验证

所有模型都通过了测试验证：
- ✅ 模型创建成功
- ✅ 前向传播正常
- ✅ 输出形状正确
- ✅ 参数冻结功能正常

运行测试脚本验证：
```bash
python test_new_models.py
```

## 🎉 总结

现在您有了一个完整的分割模型训练框架，支持10种不同的模型架构，可以根据具体需求选择最适合的模型进行训练！
