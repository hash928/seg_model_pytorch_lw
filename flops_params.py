import torch
from thop import profile

from model_utils import create_model


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
INPUT_SHAPE = (1, 3, 352, 352)
NUM_CLASSES = 2


MODEL_CONFIGS = [
    ("DeepLabV3+-Xception", lambda: create_model("deeplabv3p_xception", NUM_CLASSES)),
    ("UNet_base", lambda: create_model("unet_base", NUM_CLASSES)),
    ("DeconvNet", lambda: create_model("deconvnet", NUM_CLASSES)),
    ("SegNet", lambda: create_model("segnet", NUM_CLASSES)),
    ("PSPNet", lambda: create_model("pspnet_resnet50", NUM_CLASSES, pretrained=False)),
    ("UNet-ResNet34", lambda: create_model("unet_resnet34", NUM_CLASSES, pretrained=False)),
]


def format_metrics(flops, params):
    """格式化计算指标。"""
    return flops / 1e9, params / 1e6


def compute_model_metrics(model_name, model_builder, input_tensor):
    """计算单个模型的 FLOPs 和参数量。"""
    model = model_builder().to(DEVICE)
    model.eval()

    with torch.no_grad():
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    flops_g, params_m = format_metrics(flops, params)
    return {
        "name": model_name,
        "params_m": params_m,
        "flops_g": flops_g,
    }


def main():
    input_tensor = torch.zeros(INPUT_SHAPE).to(DEVICE)

    print(f"使用设备: {DEVICE}")
    print(f"输入尺寸: {INPUT_SHAPE}")
    print("-" * 60)

    results = []
    for model_name, model_builder in MODEL_CONFIGS:
        result = compute_model_metrics(model_name, model_builder, input_tensor)
        results.append(result)
        print(
            f"{result['name']:<20} | 参数量: {result['params_m']:>8.2f} M | "
            f"FLOPs: {result['flops_g']:>8.2f} G"
        )

    print("-" * 60)
    print("汇总完成")


if __name__ == "__main__":
    main()
