import torch
import torch.nn as nn
import torch.nn.functional as F


class DeconvNet(nn.Module):
    """
    简化版 DeconvNet：
    - encoder：与 VGG16 类似的卷积 + 最大池化（保存 indices）
    - decoder：使用 ConvTranspose2d + MaxUnpool2d 对称还原
    """

    def __init__(self, n_channels: int = 3, n_classes: int = 1):
        super(DeconvNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder blocks
        self.enc1 = self._make_block(n_channels, 64, num_convs=2)
        self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.enc2 = self._make_block(64, 128, num_convs=2)
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.enc3 = self._make_block(128, 256, num_convs=3)
        self.pool3 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.enc4 = self._make_block(256, 512, num_convs=3)
        self.pool4 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.enc5 = self._make_block(512, 512, num_convs=3)
        self.pool5 = nn.MaxPool2d(2, stride=2, return_indices=True)

        # Decoder blocks（保证各层在进入 MaxUnpool2d 前的通道数
        # 与对应 encoder 池化前通道数一致：64, 128, 256, 512, 512）
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.dec5 = self._make_block(512, 512, num_convs=3)

        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.dec4 = self._make_block(512, 256, num_convs=3)  # 输出 256 -> 对应 pool3

        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.dec3 = self._make_block(256, 128, num_convs=3)  # 输出 128 -> 对应 pool2

        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.dec2 = self._make_block(128, 64, num_convs=2)  # 输出 64 -> 对应 pool1

        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.dec1 = self._make_block(64, 64, num_convs=2)

        self.classifier = nn.Conv2d(64, n_classes, kernel_size=1)

        self._init_weights()

    @staticmethod
    def _make_block(in_channels: int, out_channels: int, num_convs: int) -> nn.Sequential:
        layers = []
        for i in range(num_convs):
            conv_in = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(conv_in, out_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        # Encoder
        x = self.enc1(x)
        x1, idx1 = self.pool1(x)

        x = self.enc2(x1)
        x2, idx2 = self.pool2(x)

        x = self.enc3(x2)
        x3, idx3 = self.pool3(x)

        x = self.enc4(x3)
        x4, idx4 = self.pool4(x)

        x = self.enc5(x4)
        x5, idx5 = self.pool5(x)

        # Decoder
        x = self.unpool5(x5, idx5, output_size=x4.shape)
        x = self.dec5(x)

        x = self.unpool4(x, idx4, output_size=x3.shape)
        x = self.dec4(x)

        x = self.unpool3(x, idx3, output_size=x2.shape)
        x = self.dec3(x)

        x = self.unpool2(x, idx2, output_size=x1.shape)
        x = self.dec2(x)

        x = self.unpool1(x, idx1)
        x = self.dec1(x)

        x = self.classifier(x)

        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)

        return x


if __name__ == "__main__":
    with torch.no_grad():
        model = DeconvNet(n_channels=3, n_classes=1).cuda()
        inp = torch.randn(1, 3, 352, 352).cuda()
        out = model(inp)
        print(out.shape)

