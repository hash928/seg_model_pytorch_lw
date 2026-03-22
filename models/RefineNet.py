import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


class _CRPBlock(nn.Module):
    """简化版 Chained Residual Pooling，用于 RefineNet"""

    def __init__(self, in_channels: int, out_channels: int, n_stages: int = 4):
        super(_CRPBlock, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1, bias=False)
             for i in range(n_stages)]
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        path = x
        for conv in self.convs:
            path = self.relu(conv(F.max_pool2d(path, kernel_size=5, stride=1, padding=2)))
            x = x + path
        return x


class _RefineBlock(nn.Module):
    """单个 RefineNet 模块：接收上一级特征 + 当前级特征"""

    def __init__(self, in_channels_low: int, in_channels_high: int, out_channels: int):
        super(_RefineBlock, self).__init__()
        self.conv_low = nn.Conv2d(in_channels_low, out_channels, kernel_size=1, bias=False)
        self.conv_high = nn.Conv2d(in_channels_high, out_channels, kernel_size=1, bias=False)
        self.crp = _CRPBlock(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, low_res: torch.Tensor, high_res: torch.Tensor) -> torch.Tensor:
        low = self.conv_low(low_res)
        high = self.conv_high(high_res)

        size = high.shape[2:]
        low = F.interpolate(low, size=size, mode="bilinear", align_corners=True)

        x = self.relu(low + high)
        x = self.crp(x)
        return x


class RefineNetResNet50(nn.Module):
    """
    基于 ResNet50 的简化 RefineNet：
    - 使用标准 torchvision resnet50，不修改主干
    - 对 layer2/3/4 的特征做逐级 Refine
    """

    def __init__(self, n_classes: int = 1, pretrained: bool = True):
        super(RefineNetResNet50, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        # 保持 backbone 结构不变
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 通道数：C2=256, C3=512, C4=1024, C5=2048
        self.refine43 = _RefineBlock(1024, 2048, 256)
        self.refine32 = _RefineBlock(512, 256, 256)
        # c2 = layer1 输出 256 通道；r3 经 refine 后也是 256 通道（不是 stem 的 64）
        self.refine21 = _RefineBlock(256, 256, 256)

        self.out_conv = nn.Conv2d(256, n_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        # Backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = x
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Refine stages
        r4 = self.refine43(c4, c5)
        r3 = self.refine32(c3, r4)
        r2 = self.refine21(c2, r3)

        out = self.out_conv(r2)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=True)
        return out


if __name__ == "__main__":
    with torch.no_grad():
        model = RefineNetResNet50(n_classes=1).cuda()
        inp = torch.randn(1, 3, 352, 352).cuda()
        out = model(inp)
        print(out.shape)

