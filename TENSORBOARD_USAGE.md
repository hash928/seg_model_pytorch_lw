# TensorBoard 使用指南

## 模块化设计

TensorBoard 功能已经被模块化到 `tensorboard_logger.py` 中，使代码更易维护。

## 使用方法

### 1. 开始训练
```bash
python train_unet.py
```

### 2. 启动 TensorBoard
```bash
tensorboard --logdir=./checkpoints/your_model_training/runs --port=6006
```

### 3. 查看可视化
打开浏览器访问：`http://localhost:6006`

## TensorBoard 功能

### SCALARS 标签页
- **Train/Loss**: 训练损失曲线
- **Train/IoU**: 训练 IoU 曲线  
- **Train/Dice**: 训练 Dice 曲线
- **Val/Loss**: 验证损失曲线
- **Val/IoU**: 验证 IoU 曲线
- **Val/Dice**: 验证 Dice 曲线
- **Learning_Rate**: 学习率变化曲线

### GRAPHS 标签页
- 完整的模型计算图
- 数据流和操作关系

### HISTOGRAMS 标签页
- **Parameters/**: 模型参数分布
- **Gradients/**: 梯度分布（每10个epoch记录）

### IMAGES 标签页
- **Sample_Predictions/**: 样本图像和预测结果（每10个epoch记录）

## 模块化优势

1. **代码分离**: TensorBoard 功能独立于训练逻辑
2. **易于维护**: 所有 TensorBoard 相关代码集中在一个文件中
3. **可重用**: 可以在其他项目中轻松复用
4. **可扩展**: 容易添加新的可视化功能
5. **可测试**: 可以独立测试 TensorBoard 功能

## 自定义扩展

如需添加新的可视化功能，只需在 `TensorBoardLogger` 类中添加新方法：

```python
def log_custom_metric(self, metric_name, value, epoch):
    """记录自定义指标"""
    self.writer.add_scalar(f"Custom/{metric_name}", value, epoch)
```
