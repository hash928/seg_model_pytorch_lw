import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

"""
Modified Aligned Xception

(1) deeper Xception same as in [31] except that we do not modify the entry flow 
network structure for fast computation and memory efficiency

更深的网路， Middle Flow 重复16个block

(2) all max pooling operations are replaced by depthwise separable
convolution with striding, which enables us to apply atrous separable convolu-
tion to extract feature maps at an arbitrary resolution (another option is to
extend the atrous algorithm to max pooling operations)

将Maxpool下采样改为带有stride的深度可分离卷积SepConv2d

(3) extra batch normalization [75] and ReLU activation are added after each 3x3 
depthwise convolution, similar to MobileNet design [29]



"""

def _make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    将通道数调整为可整除的整数，便于加速与对齐。
    参考 MobileNet 的 make_divisible 思路。
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 避免四舍五入导致下降过多
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


def _print_shape(func):
    """
    测试打印输入输出维度
    :param func:
    :return:
    """

    def print_shape(*args, **kwargs):
        res = func(*args, **kwargs)
        print(func, res.shape)
        return res

    return print_shape


class SepConv2d(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3, stride=1,
                 dilation=1, bias=False):
        """
        深度可分离卷积
        第一个卷积depthwise_conv在spatial上，每个channel单独进行，用group=in_planes
        第二个卷积pointwise_conv在cross-channel上，相当于用1x1卷积调整维度
        :param in_planes: in_channels
        :param planes: out_channels
        :param kernel_size: kernel_size
        :param stride: depthwise conv的stride
        :param dilation: depthwise conv的padding
        :param bias: bias，因为后面都接BN，默认False
        """
        super(SepConv2d, self).__init__()
        padding = (kernel_size - 1) // 2 * dilation  # 都是same卷积
        self.depthwise_conv = nn.Conv2d(
            in_planes, in_planes, kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=in_planes, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_planes, planes, 1, bias=bias)
        pass

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

    pass


class Block(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dilation=1):
        """
        Entry Flow和Middlw Flow用的Block
        只有第三个卷积来决定是否下采样
        :param in_planes: in_planes
        :param planes: planes
        :param stride: stride决定第三个卷积是否下采样
        :param dilation: dilation
        """
        super(Block, self).__init__()
        self.conv1 = nn.Sequential(
            SepConv2d(in_planes, planes, 3, stride=1, dilation=dilation),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            SepConv2d(planes, planes, 3, stride=1, dilation=dilation),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            SepConv2d(planes, planes, 3, stride=stride, dilation=dilation),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))  # 第三个conv的stride决定是否下采样
        self.project = None
        if in_planes != planes or stride != 1:
            self.project = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride),
                nn.BatchNorm2d(planes))  # residual connection
        pass

    # @_print_shape
    def forward(self, x):
        identity = x  # residual connection 准备
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.project is not None:
            identity = self.project(identity)  # residual connection 相加
        x = x + identity
        return F.relu(x, inplace=True)

    pass


class ExitBlock(nn.Module):
    def __init__(self, in_planes=728, planes=1024, stride=1, dilation=1):
        """
        Exit Flow用的Block
        728->728
        728->1024
        1024->1024
        :param in_planes: in_planes=728
        :param planes: planes=1024
        :param stride: stride决定第三个卷积是否下采样
        :param dilation: dilation
        """
        super(ExitBlock, self).__init__()
        self.conv1 = nn.Sequential(
            SepConv2d(in_planes, in_planes, 3, stride=1, dilation=dilation),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            SepConv2d(in_planes, planes, 3, stride=1, dilation=dilation),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            SepConv2d(planes, planes, 3, stride=stride, dilation=dilation),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))  # 第三个conv的stride决定是否下采样
        self.project = None
        if in_planes != planes or stride != 1:
            self.project = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride),
                nn.BatchNorm2d(planes))
        pass

    # @_print_shape
    def forward(self, x):
        identity = x  # residual connection 准备
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.project is not None:
            identity = self.project(identity)  # residual connection 相加
        x = x + identity
        return x

    pass


class XceptionBackbone(nn.Module):
    def __init__(self, in_planes, output_stride=16, width_mult: float = 1.0):
        """
        用于DeepLabV3+的AlignedXception
        :param in_planes: 输入通道，也就是图像的通道
        :param output_stride: 主干输出spatial与输入spatial的比值可以是8,16,32,论文采用16最好
        :param width_mult: 宽度系数，用于按比例缩放通道数（例如 0.75/0.5）。
        """
        super(XceptionBackbone, self).__init__()
        # 根据DeepLabV3讨论的Atrous Conv
        # strides[0]和dilations[0] 在Entry Flow的第三个block使用
        # Middle Flow都不进行下采样，stride都等于1，dilation使用dilations[1]
        # strides[1]和dilations[1] 在Exit Flow的block使用
        if output_stride == 8:  # os=8时，最后一次下采样dilation=1，之后dilation=4
            strides = (1, 1)
            dilations = (4, 4)
        elif output_stride == 16:  # os=16时，最后一次下采样dilation=1，之后dilation=2
            strides = (2, 1)
            dilations = (1, 2)
        elif output_stride == 32:  # os=32时，最后一次下采样dilation=1，之后dilation=1
            strides = (2, 2)
            dilations = (1, 1)
        else:
            raise ValueError('output stride error!')

        if width_mult <= 0:
            raise ValueError('width_mult must be > 0')

        c32 = _make_divisible(32 * width_mult)
        c64 = _make_divisible(64 * width_mult)
        c128 = _make_divisible(128 * width_mult)
        c256 = _make_divisible(256 * width_mult)
        c728 = _make_divisible(728 * width_mult)
        c1024 = _make_divisible(1024 * width_mult)
        c1536 = _make_divisible(1536 * width_mult)
        c2048 = _make_divisible(2048 * width_mult)

        # Entry Flow
        self.entry_conv1 = nn.Sequential(
            nn.Conv2d(in_planes, c32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c32),
            nn.ReLU(inplace=True))
        self.entry_conv2 = nn.Sequential(
            nn.Conv2d(c32, c64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c64),
            nn.ReLU(inplace=True))
        self.entry_block1 = nn.Sequential(Block(c64, c128, stride=2))
        self.entry_block2 = nn.Sequential(Block(c128, c256, stride=2))
        self.entry_block3 = nn.Sequential(
            Block(c256, c728, stride=strides[0], dilation=dilations[0]))

        # Middle Flow
        mid_blocks = [Block(c728, c728, stride=1, dilation=dilations[1])] * 16
        self.mid_blocks = nn.Sequential(*mid_blocks)

        # Exit Flow
        self.exit_block = ExitBlock(c728, c1024, stride=strides[1], dilation=dilations[1])
        self.exit_conv1 = nn.Sequential(
            SepConv2d(c1024, c1536, 3, dilation=dilations[1]),
            nn.BatchNorm2d(c1536),
            nn.ReLU(inplace=True))
        self.exit_conv2 = nn.Sequential(
            SepConv2d(c1536, c1536, 3, dilation=dilations[1]),
            nn.BatchNorm2d(c1536),
            nn.ReLU(inplace=True))
        self.exit_conv3 = nn.Sequential(
            SepConv2d(c1536, c2048, 3, dilation=dilations[1]),
            nn.BatchNorm2d(c2048),
            nn.ReLU(inplace=True))
        self._init_weight()
        pass

    def _init_weight(self):
        """
        初始化参数
        :return:
        """
        pass

    def forward(self, x):
        # Entry Flow
        x = self.entry_conv1(x)  # 2x
        x = self.entry_conv2(x)  # 2x
        x = self.entry_block1(x)  # 4x
        low_level_features = x  # low-level feature
        x = self.entry_block2(x)  # 8x
        x = self.entry_block3(x)  # os=8,8x|os=16,16x|os=32,16x

        # Middle Flow
        x = self.mid_blocks(x)  # os=8,8x|os=16,16x|os=32,16x

        # Exit Flow
        x = self.exit_block(x)  # os=8,8x|os=16,16x|os=32,32x 此后不再下采样
        x = self.exit_conv1(x)
        x = self.exit_conv2(x)
        x = self.exit_conv3(x)

        return x, low_level_features

    pass


def xception_backbone(in_channels, output_stride=16, width_mult: float = 1.0):
    if output_stride in (8, 16, 32):
        return XceptionBackbone(in_channels, output_stride, width_mult=width_mult)
    else:
        raise ValueError('output stride error! should be 8, 16 or 32')


if __name__ == '__main__':
    batch_size = 1
    in_dims = 3
    num_class = 8
    im = torch.randint(0, 256, size=(batch_size, in_dims, 299, 299),
                       dtype=torch.float, requires_grad=True)

    print(im.shape)

    model = xception_backbone(in_dims, 16)
    output, low_level = model(im)
    print(output.shape, low_level.shape)

    lb = torch.randint(0, num_class,
                       size=(output.shape[0], output.shape[2], output.shape[3]),
                       dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters())
    loss = F.cross_entropy(output, lb)
    loss.backward()
    optimizer.step()
    print(loss.detach().item())
    pass