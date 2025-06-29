# SAM2 训练和测试工具

这是一个用于训练和测试 SAM2 (Segment Anything Model 2) 的完整工具包。

## 文件结构

```
.
├── train_sam2.py          # 主训练脚本
├── dataset.py             # 数据集加载器
├── test_single_image.py   # 单张图像测试脚本
├── test_batch.py          # 批量测试脚本
├── requirements.txt       # 依赖包列表
└── README.md             # 使用说明
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (推荐)
- 至少 16GB GPU 内存

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据准备

### 数据格式

数据目录结构应该如下：

```
data_dir/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── masks/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── annotations.json
```

### 标注文件格式

`annotations.json` 文件应该包含以下格式的标注：

```json
{
  "image1.jpg": {
    "points": [[x1, y1], [x2, y2], ...],
    "labels": [1, 1, ...],
    "mask_path": "masks/image1.png"
  },
  "image2.jpg": {
    "points": [[x1, y1], [x2, y2], ...],
    "labels": [1, 1, ...],
    "mask_path": "masks/image2.png"
  }
}
```

其中：
- `points`: 标注点的坐标列表 `[x, y]`
- `labels`: 对应的标签列表，1表示前景点，0表示背景点
- `mask_path`: 对应的掩码图像路径

## 训练模型

### 基本训练命令

```bash
python train_sam2.py \
    --data_dir /path/to/your/data \
    --output_dir ./outputs \
    --epochs 100 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4
```

### 训练参数说明

- `--data_dir`: 数据目录路径
- `--output_dir`: 输出目录路径
- `--epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--learning_rate`: 学习率
- `--weight_decay`: 权重衰减
- `--num_workers`: 数据加载器工作进程数
- `--save_interval`: 模型保存间隔（轮数）
- `--eval_interval`: 评估间隔（轮数）
- `--resume`: 从检查点恢复训练

### 训练监控

训练过程中会显示：
- 当前轮数和进度
- 训练损失和验证损失
- 学习率变化
- GPU 内存使用情况
- 训练速度

### 模型保存

- 最佳模型保存在 `{output_dir}/best_model.pth`
- 最新模型保存在 `{output_dir}/latest_model.pth`
- 训练日志保存在 `{output_dir}/training_log.txt`

## 测试模型

### 单张图像测试

```bash
python test_single_image.py \
    --model_path ./outputs/best_model.pth \
    --image_path /path/to/test/image.jpg \
    --points "100,200;300,400" \
    --save_path ./test_result.png
```

参数说明：
- `--model_path`: 训练好的模型路径
- `--image_path`: 测试图像路径
- `--points`: 输入点坐标，格式为 "x1,y1;x2,y2"
- `--save_path`: 结果保存路径（可选）

### 批量测试

```bash
python test_batch.py \
    --model_path ./outputs/best_model.pth \
    --data_dir /path/to/test/data \
    --save_dir ./test_results \
    --num_samples 100
```

参数说明：
- `--model_path`: 训练好的模型路径
- `--data_dir`: 测试数据目录
- `--save_dir`: 结果保存目录
- `--num_samples`: 测试样本数量限制（可选）

### 评估指标

批量测试会计算以下指标：
- **IoU (Intersection over Union)**: 交并比
- **Dice系数**: 相似度度量
- **精确率 (Precision)**: 预测为正例中实际为正例的比例
- **召回率 (Recall)**: 实际正例中被预测为正例的比例
- **F1分数**: 精确率和召回率的调和平均
- **准确率 (Accuracy)**: 正确预测的比例

## 使用示例

### 1. 准备数据

```bash
# 创建数据目录
mkdir -p data/images data/masks

# 将图像放入 images 目录
cp your_images/* data/images/

# 将掩码放入 masks 目录
cp your_masks/* data/masks/

# 创建标注文件
python create_annotations.py  # 需要自己实现
```

### 2. 开始训练

```bash
python train_sam2.py \
    --data_dir ./data \
    --output_dir ./outputs \
    --epochs 50 \
    --batch_size 2 \
    --learning_rate 5e-5
```

### 3. 监控训练

```bash
# 查看训练日志
tail -f ./outputs/training_log.txt

# 查看GPU使用情况
nvidia-smi
```

### 4. 测试模型

```bash
# 单张图像测试
python test_single_image.py \
    --model_path ./outputs/best_model.pth \
    --image_path ./test_image.jpg \
    --points "500,300;600,400" \
    --save_path ./result.png

# 批量测试
python test_batch.py \
    --model_path ./outputs/best_model.pth \
    --data_dir ./test_data \
    --save_dir ./test_results
```

## 性能优化建议

### 1. 数据加载优化
- 使用SSD存储数据
- 增加 `num_workers` 参数
- 使用 `pin_memory=True`

### 2. 训练优化
- 根据GPU内存调整 `batch_size`
- 使用混合精度训练
- 使用梯度累积

### 3. 模型优化
- 使用预训练权重
- 调整学习率调度
- 使用数据增强

## 常见问题

### 1. 内存不足
- 减小 `batch_size`
- 减小图像尺寸
- 使用梯度累积

### 2. 训练速度慢
- 检查数据加载速度
- 使用更快的存储设备
- 增加 `num_workers`

### 3. 模型不收敛
- 检查学习率设置
- 检查数据质量
- 调整损失函数权重

### 4. 预测效果差
- 增加训练数据
- 调整模型参数
- 检查数据标注质量

## 技术支持

如果遇到问题，请检查：
1. 数据格式是否正确
2. 依赖包版本是否兼容
3. GPU驱动是否最新
4. 内存是否充足

## 许可证

本项目基于 MIT 许可证开源。