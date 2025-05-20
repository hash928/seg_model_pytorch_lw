import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet34_Weights


class FCN(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, backbone='resnet50', pretrained=True):
        super(FCN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # 加载预训练的backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            self.in_channels = 2048
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            self.in_channels = 512
        else:
            raise ValueError(f"不支持的backbone: {backbone}")
        
        # 移除最后的全连接层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # FCN头部
        self.fcn_head = nn.Sequential(
            nn.Conv2d(self.in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, n_classes, kernel_size=1)
        )
        
        # 上采样层
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(n_classes, n_classes, kernel_size=32, stride=32, padding=0, bias=False)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.fcn_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 获取backbone特征
        features = self.backbone(x)
        
        # FCN头部
        x = self.fcn_head(features)
        
        # 上采样到原始大小
        x = self.upsample(x)
        
        # 确保输出大小与输入一致
        if x.shape[2:] != x.shape[2:]:
            x = F.interpolate(x, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return x


if __name__ == "__main__":
    with torch.no_grad():
        model = FCN(backbone='resnet50').cuda()
        x = torch.randn(1, 3, 352, 352).cuda()
        out = model(x)
        print(out.shape) 