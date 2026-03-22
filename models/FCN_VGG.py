import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights


class FCN32s(nn.Module):
    """
    原始 FCN-32s 结构，使用 VGG16 backbone：
    - 直接在最后一层特征上做 1x1 卷积分类
    - 再一次性上采样到输入尺寸
    """

    def __init__(self, n_channels: int = 3, n_classes: int = 1, pretrained: bool = True):
        super(FCN32s, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        features = list(vgg.features.children())

        # 使用 VGG16 全部特征层（到 pool5）
        self.backbone = nn.Sequential(*features)

        # 将原来的 FC 层改为卷积层
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.score_fr = nn.Conv2d(4096, n_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.fc6, self.fc7, self.score_fr]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        x = self.backbone(x)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        x = self.score_fr(x)

        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)
        return x


if __name__ == "__main__":
    with torch.no_grad():
        model = FCN32s(n_channels=3, n_classes=1).cuda()
        inp = torch.randn(1, 3, 352, 352).cuda()
        out = model(inp)
        print(out.shape)

