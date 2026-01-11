from thop import profile
import torch
from model_utils import create_model
from models import UNetb

device = torch.device('cuda:0')
# UNet_base模型参数计算
model0 = UNetb.unet_base(3, 2)

# 其他模型参数计算
model1 = create_model("unet_base", 2, "false", "false",
                     "false", "false", "false")

# UNetplusplus模型参数计算

model =model0.to(device)
input = torch.zeros((1, 3, 352, 352)).to(device)
# 单位转换和格式化输出
def format_metrics(flops, params):
    """格式化计算指标"""
    params_m = params / 1e6
    flops_g = flops / 1e9
    return flops_g, params_m
flops, params = profile(model.to(device), inputs=(input,))
# 获取格式化后的值
flops_g, params_m = format_metrics(flops, params)

# 格式化输出
print(f"参数量: {params_m:.2f} M")
print(f"FLOPs: {flops_g:.2f} G")

