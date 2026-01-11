# UNetb Grad-CAM 热力图可视化使用指南

本工具使用 `pytorch-grad-cam` 为 UNetb 模型生成 Grad-CAM 热力图可视化，帮助理解模型在分割任务中的关注区域。

## 安装依赖

首先需要安装 `pytorch-grad-cam`：

```bash
pip install grad-cam
```

## 使用方法

### 基本用法

```bash
python visualize_gradcam.py \
    --image_path "/path/to/your/image.png" \
    --checkpoint "/path/to/your/checkpoint.pth" \
    --model_type "unet_resnet34" \
    --num_classes 1 \
    --use_aspp \
    --attention_type "se"
```

### 参数说明

- `--image_path`: **必需**，输入图像路径
- `--checkpoint`: 模型权重路径（可选，如果不提供则使用随机初始化的模型）
- `--model_type`: 模型类型，可选值：
  - `unet_base`: 基础 UNet
  - `unet_resnet18`, `unet_resnet34`, `unet_resnet50`, `unet_resnet101`, `unet_resnet152`: 使用 ResNet 作为编码器的 UNet
- `--num_classes`: 分割类别数（默认：1，二值分割）
- `--use_aspp`: 是否使用 ASPP 模块
- `--attention_type`: 注意力模块类型，可选值：`se`, `cbam`, `ca`, `eca`
- `--target_class`: 目标类别（对于多类别分割，默认：0）
- `--layer_name`: 目标层名称，可选值：
  - `encoder`: 编码器最后一层（推荐用于查看编码器关注区域）
  - `decoder`: 解码器第一层
  - `bottleneck`: bottleneck 层（ASPP 或注意力模块）
- `--cam_method`: CAM 方法，可选值：
  - `gradcam`: 标准 Grad-CAM（默认）
  - `gradcam++`: Grad-CAM++
  - `xgradcam`: XGrad-CAM
  - `eigencam`: EigenCAM
- `--save_path`: 保存路径（可选，如果不提供则显示图像）
- `--alpha`: 热力图透明度，范围 0-1（默认：0.4）

### 示例

#### 示例 1: 使用编码器层生成热力图

```bash
python visualize_gradcam.py \
    --image_path "test_image.png" \
    --checkpoint "checkpoints/model.pth" \
    --model_type "unet_resnet34" \
    --num_classes 1 \
    --use_aspp \
    --attention_type "se" \
    --layer_name "encoder" \
    --cam_method "gradcam" \
    --save_path "result_encoder.png"
```

#### 示例 2: 使用 bottleneck 层和 Grad-CAM++

```bash
python visualize_gradcam.py \
    --image_path "test_image.png" \
    --checkpoint "checkpoints/model.pth" \
    --model_type "unet_resnet34" \
    --num_classes 1 \
    --use_aspp \
    --attention_type "se" \
    --layer_name "bottleneck" \
    --cam_method "gradcam++" \
    --alpha 0.5 \
    --save_path "result_bottleneck.png"
```

#### 示例 3: 多类别分割

```bash
python visualize_gradcam.py \
    --image_path "test_image.png" \
    --checkpoint "checkpoints/model.pth" \
    --model_type "unet_resnet34" \
    --num_classes 3 \
    --target_class 1 \
    --layer_name "encoder" \
    --save_path "result_class1.png"
```

## 输出说明

脚本会生成一个包含三张子图的图像：

1. **原始图像**: 输入的原始图像
2. **Grad-CAM 热力图**: 纯热力图，颜色从蓝色（低激活）到红色（高激活）
3. **叠加结果**: 热力图叠加在原始图像上的结果

热力图中：
- **红色/橙色区域**: 模型高度关注的区域（高激活）
- **黄色/绿色区域**: 中等关注区域
- **蓝色区域**: 低关注区域

## 注意事项

1. **模型配置匹配**: 确保 `--model_type`, `--use_aspp`, `--attention_type` 等参数与训练时使用的配置一致
2. **类别数匹配**: `--num_classes` 必须与训练时的类别数一致
3. **权重加载**: 如果权重文件中的键名与模型不匹配（例如 `decoder.se.*` vs `decoder.attention.*`），脚本会自动处理
4. **目标层选择**: 
   - `encoder`: 适合查看模型在编码阶段关注的特征
   - `bottleneck`: 适合查看经过 ASPP 或注意力模块后的特征
   - `decoder`: 适合查看解码阶段的特征

## 在代码中使用

你也可以在 Python 代码中直接使用：

```python
from models.UNetb import unet_resnet
from visualize_gradcam import visualize_gradcam

# 创建模型
model = unet_resnet('resnet34', 3, 1, pretrained=False, 
                    use_aspp=True, attention_type='se')

# 生成可视化
visualize_gradcam(
    model=model,
    image_path="test_image.png",
    checkpoint_path="checkpoints/model.pth",
    target_class=0,
    layer_name='encoder',
    cam_method='gradcam',
    save_path="result.png"
)
```

## 故障排除

1. **导入错误**: 确保已安装 `pytorch-grad-cam`: `pip install grad-cam`
2. **CUDA 错误**: 如果遇到 CUDA 相关错误，脚本会自动回退到 CPU
3. **权重加载失败**: 检查权重文件路径和模型配置是否匹配
4. **层找不到**: 确保 `--layer_name` 与模型结构匹配（例如，如果模型没有使用 ASPP，不要选择 `bottleneck`）
