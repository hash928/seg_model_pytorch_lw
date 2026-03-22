import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


class _PyramidPoolingModule(nn.Module):
    """Pyramid Pooling Module used in PSPNet"""

    def __init__(self, in_channels: int, pool_sizes=(1, 2, 3, 6)):
        super(_PyramidPoolingModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        self.paths = nn.ModuleList()
        for ps in pool_sizes:
            self.paths.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(ps),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        outputs = [x]
        for path in self.paths:
            out = path(x)
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
            outputs.append(out)
        return torch.cat(outputs, dim=1)


class PSPNetResNet50(nn.Module):
    """
    基于 ResNet50 的简化 PSPNet：
    - 标准 torchvision resnet50 作为 backbone（不改结构）
    - 使用 Pyramid Pooling Module 做多尺度上下文聚合
    """

    def __init__(self, n_classes: int = 1, pretrained: bool = True):
        super(PSPNetResNet50, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        # backbone 前四个 stage
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        in_channels = 2048
        self.ppm = _PyramidPoolingModule(in_channels=in_channels, pool_sizes=(1, 2, 3, 6))

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, n_classes, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.final_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.ppm(x)
        x = self.final_conv(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)
        return x


if __name__ == "__main__":
    with torch.no_grad():
        model = PSPNetResNet50(n_classes=1).cuda()
        inp = torch.randn(1, 3, 352, 352).cuda()
        out = model(inp)
        print(out.shape)

