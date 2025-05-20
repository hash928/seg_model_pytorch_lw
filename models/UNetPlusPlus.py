import torch
import torch.nn as nn
import torch.nn.functional as F
from .UNet import DoubleConv

class UNetPlusPlus(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, deep_supervision=False):
        super(UNetPlusPlus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # Decoder with nested skip connections
        self.up1_0 = DoubleConv(1024 + 512, 512)
        self.up2_0 = DoubleConv(512 + 256, 256)
        self.up3_0 = DoubleConv(256 + 128, 128)
        self.up4_0 = DoubleConv(128 + 64, 64)

        self.up1_1 = DoubleConv(512 + 256, 256)
        self.up2_1 = DoubleConv(256 + 128, 128)
        self.up3_1 = DoubleConv(128 + 64, 64)

        self.up1_2 = DoubleConv(256 + 128, 128)
        self.up2_2 = DoubleConv(128 + 64, 64)

        self.up1_3 = DoubleConv(128 + 64, 64)

        # Output layers
        self.outc0 = nn.Conv2d(64, n_classes, kernel_size=1)
        if deep_supervision:
            self.outc1 = nn.Conv2d(64, n_classes, kernel_size=1)
            self.outc2 = nn.Conv2d(64, n_classes, kernel_size=1)
            self.outc3 = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1_0 = self.inc(x)
        x2_0 = self.down1(x1_0)
        x3_0 = self.down2(x2_0)
        x4_0 = self.down3(x3_0)
        x5_0 = self.down4(x4_0)

        # Decoder with nested skip connections
        x1_1 = F.interpolate(x5_0, size=x4_0.shape[2:], mode='bilinear', align_corners=True)
        x1_1 = self.up1_0(torch.cat([x1_1, x4_0], dim=1))

        x2_1 = F.interpolate(x1_1, size=x3_0.shape[2:], mode='bilinear', align_corners=True)
        x2_1 = self.up2_0(torch.cat([x2_1, x3_0], dim=1))

        x3_1 = F.interpolate(x2_1, size=x2_0.shape[2:], mode='bilinear', align_corners=True)
        x3_1 = self.up3_0(torch.cat([x3_1, x2_0], dim=1))

        x4_1 = F.interpolate(x3_1, size=x1_0.shape[2:], mode='bilinear', align_corners=True)
        x4_1 = self.up4_0(torch.cat([x4_1, x1_0], dim=1))

        x1_2 = F.interpolate(x1_1, size=x3_0.shape[2:], mode='bilinear', align_corners=True)
        x1_2 = self.up1_1(torch.cat([x1_2, x3_0], dim=1))

        x2_2 = F.interpolate(x1_2, size=x2_0.shape[2:], mode='bilinear', align_corners=True)
        x2_2 = self.up2_1(torch.cat([x2_2, x2_0], dim=1))

        x3_2 = F.interpolate(x2_2, size=x1_0.shape[2:], mode='bilinear', align_corners=True)
        x3_2 = self.up3_1(torch.cat([x3_2, x1_0], dim=1))

        x1_3 = F.interpolate(x1_2, size=x2_0.shape[2:], mode='bilinear', align_corners=True)
        x1_3 = self.up1_2(torch.cat([x1_3, x2_0], dim=1))

        x2_3 = F.interpolate(x1_3, size=x1_0.shape[2:], mode='bilinear', align_corners=True)
        x2_3 = self.up2_2(torch.cat([x2_3, x1_0], dim=1))

        x1_4 = F.interpolate(x1_3, size=x1_0.shape[2:], mode='bilinear', align_corners=True)
        x1_4 = self.up1_3(torch.cat([x1_4, x1_0], dim=1))

        if self.deep_supervision:
            output1 = self.outc0(x4_1)
            output2 = self.outc1(x3_2)
            output3 = self.outc2(x2_3)
            output4 = self.outc3(x1_4)
            return [output1, output2, output3, output4]
        else:
            output = self.outc0(x1_4)
            return output


class UNetPlus(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNetPlus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # Decoder with dense skip connections
        self.up1 = DoubleConv(1024 + 512 + 256 + 128 + 64, 512)
        self.up2 = DoubleConv(512 + 256 + 128 + 64, 256)
        self.up3 = DoubleConv(256 + 128 + 64, 128)
        self.up4 = DoubleConv(128 + 64, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # 保存输入尺寸用于最终上采样
        input_size = x.size()[2:]
        
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with dense skip connections
        # 调整所有特征图到相同的空间尺寸
        target_size = x5.shape[2:]
        x4_resized = F.interpolate(x4, size=target_size, mode='bilinear', align_corners=True)
        x3_resized = F.interpolate(x3, size=target_size, mode='bilinear', align_corners=True)
        x2_resized = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=True)
        x1_resized = F.interpolate(x1, size=target_size, mode='bilinear', align_corners=True)
        
        x = self.up1(torch.cat([x5, x4_resized, x3_resized, x2_resized, x1_resized], dim=1))
        
        # 调整特征图尺寸用于下一层
        target_size = x.shape[2:]
        x3_resized = F.interpolate(x3, size=target_size, mode='bilinear', align_corners=True)
        x2_resized = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=True)
        x1_resized = F.interpolate(x1, size=target_size, mode='bilinear', align_corners=True)
        
        x = self.up2(torch.cat([x, x3_resized, x2_resized, x1_resized], dim=1))
        
        # 调整特征图尺寸用于下一层
        target_size = x.shape[2:]
        x2_resized = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=True)
        x1_resized = F.interpolate(x1, size=target_size, mode='bilinear', align_corners=True)
        
        x = self.up3(torch.cat([x, x2_resized, x1_resized], dim=1))
        
        # 调整特征图尺寸用于最后一层
        target_size = x.shape[2:]
        x1_resized = F.interpolate(x1, size=target_size, mode='bilinear', align_corners=True)
        
        x = self.up4(torch.cat([x, x1_resized], dim=1))

        # 生成输出
        logits = self.outc(x)
        
        # 将输出上采样到输入图像的尺寸
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=True)
        
        return logits

if __name__ == "__main__":
    with torch.no_grad():
        # Test UNet++
        model_plusplus = UNetPlusPlus(deep_supervision=True).cuda()
        x = torch.randn(1, 3, 352, 352).cuda()
        out = model_plusplus(x)
        if isinstance(out, list):
            print("UNet++ outputs:", [o.shape for o in out])
        else:
            print("UNet++ output:", out.shape)

        # Test UNet+
        model_plus = UNetPlus().cuda()
        x = torch.randn(1, 3, 352, 352).cuda()
        out = model_plus(x)
        print("UNet+ output:", out.shape) 