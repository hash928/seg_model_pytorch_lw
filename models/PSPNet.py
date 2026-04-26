import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


class _PyramidPoolingModule(nn.Module):
    """Pyramid Pooling Module used in PSPNet."""

    def __init__(self, in_channels: int, out_channels: int = 512, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        branch_channels = out_channels // len(pool_sizes)
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
            for pool_size in pool_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        outputs = [x]
        for path in self.paths:
            pooled = path(x)
            pooled = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
            outputs.append(pooled)
        return torch.cat(outputs, dim=1)


class PSPNetResNet50(nn.Module):
    """
    更接近标准 PSPNet 的 ResNet50 版本：
    - 使用 dilated ResNet50 backbone，保留更高分辨率特征图
    - 使用 Pyramid Pooling Module 聚合多尺度上下文
    - 在 1/8 特征图上完成分类后再上采样回输入分辨率
    """

    def __init__(self, n_classes: int = 1, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights, replace_stride_with_dilation=[False, True, True])

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        in_channels = 2048
        ppm_out_channels = 512
        self.ppm = _PyramidPoolingModule(in_channels=in_channels, out_channels=ppm_out_channels, pool_sizes=(1, 2, 3, 6))

        classifier_in_channels = in_channels + ppm_out_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(classifier_in_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, n_classes, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.ppm.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        for module in self.classifier.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

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
        x = self.classifier(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


if __name__ == "__main__":
    with torch.no_grad():
        model = PSPNetResNet50(n_classes=1).cuda()
        inp = torch.randn(1, 3, 352, 352).cuda()
        out = model(inp)
        print(out.shape)
