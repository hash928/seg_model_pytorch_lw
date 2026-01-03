import torch
import torch.nn as nn
import numpy as np


'''-------------辅助函数-----------------------------'''
class h_swish(nn.Module):
    """Hard-Swish激活函数"""
    def __init__(self):
        super(h_swish, self).__init__()
    
    def forward(self, x):
        return x * nn.functional.relu6(x + 3) / 6


'''-------------SE模块-----------------------------'''
#全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
            # 读取批数据图片数量及通道数
            b, c, h, w = x.size()
            # Fsq操作：经池化后输出b*c的矩阵
            y = self.gap(x).view(b, c)
            # Fex操作：经全连接层输出（b，c，1，1）矩阵
            y = self.fc(y).view(b, c, 1, 1)
            # Fscale操作：将得到的权重乘以原来的特征图x
            return x * y.expand_as(x)

'''-------------CBAM模块-----------------------------'''
class CBAM_Block(nn.Module):

    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM_Block, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

'''-------------CA模块-----------------------------'''
class CA_Block(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CA_Block, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y

'''-------------ECA模块-----------------------------'''
class ECA_Block(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA_Block, self).__init__()
        k = int(abs((np.log2(channels) + b) / gamma))
        k = k if k % 2 else k + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # [B,C,1,1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # [B,1,C]
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))  # [B,C,1,1]
        return x * y.expand_as(x)


# ============== 测试代码 ==============
if __name__ == "__main__":
    # 1. 基础功能测试
    print("=" * 50)
    print("1. 基础功能测试")
    print("=" * 50)

    # 创建SE模块
    inchannel = 256
    se_block = SE_Block(inchannel, ratio=16)

    # 创建测试输入
    batch_size = 4
    height, width = 7, 7
    x = torch.randn(batch_size, inchannel, height, width)

    print(f"输入维度: {x.shape}")

    # 前向传播
    output = se_block(x)

    print(f"输出维度: {output.shape}")
    print(f"输入输出形状是否一致: {x.shape == output.shape}")

    # 2. 验证权重计算
    print("\n" + "=" * 50)
    print("2. 权重验证测试")
    print("=" * 50)

    # 创建一个简单的测试输入，便于观察
    test_input = torch.ones(1, inchannel, 2, 2)  # 全1输入
    test_output = se_block(test_input)

    print(f"测试输入值: 全1矩阵")
    print(f"输出范围检查: min={test_output.min():.4f}, max={test_output.max():.4f}")

    # 3. 不同参数测试
    print("\n" + "=" * 50)
    print("3. 不同参数配置测试")
    print("=" * 50)

    configs = [
        {"inchannel": 32, "ratio": 8},
        {"inchannel": 128, "ratio": 16},
        {"inchannel": 256, "ratio": 32}
    ]

    for i, config in enumerate(configs):
        inchannel = config["inchannel"]
        ratio = config["ratio"]

        se = SE_Block(inchannel, ratio)
        test_x = torch.randn(2, inchannel, 16, 16)
        test_y = se(test_x)

        print(f"配置{i + 1}: inchannel={inchannel}, ratio={ratio}")
        print(f"  输入维度: {test_x.shape}")
        print(f"  输出维度: {test_y.shape}")
        print(f"  通道缩减比: 1/{ratio}")

    # 4. 梯度检查
    print("\n" + "=" * 50)
    print("4. 梯度流测试")
    print("=" * 50)

    # 确保模型处于训练模式
    se_block.train()

    # 创建需要梯度的输入
    x_with_grad = torch.randn(batch_size, inchannel, height, width, requires_grad=True)
    target = torch.randn_like(x_with_grad)

    # 前向传播
    output = se_block(x_with_grad)

    # 计算损失和梯度
    loss = nn.MSELoss()(output, target)
    loss.backward()

    print(f"损失值: {loss.item():.6f}")
    print(f"输入梯度是否存在: {x_with_grad.grad is not None}")
    print(f"输入梯度形状: {x_with_grad.grad.shape}")

    # 检查模型参数梯度
    total_params = sum(p.numel() for p in se_block.parameters())
    params_with_grad = sum(p.numel() for p in se_block.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params}")
    print(f"需要梯度的参数: {params_with_grad}")

    # 5. 边缘情况测试
    print("\n" + "=" * 50)
    print("5. 边缘情况测试")
    print("=" * 50)

    # 测试ratio不能整除的情况
    try:
        se_edge = SE_Block(63, ratio=16)  # 63不能被16整除
        print("ratio不能整除测试: 成功创建")
    except Exception as e:
        print(f"ratio不能整除测试: 错误 - {type(e).__name__}")

    # 测试小batch size
    se_block.eval()  # 切换到评估模式
    with torch.no_grad():
        small_batch = torch.randn(1, inchannel, 8, 8)
        small_output = se_block(small_batch)
        print(f"batch_size=1测试: 输出形状 {small_output.shape}")

    # 6. 集成到神经网络中的测试
    print("\n" + "=" * 50)
    print("6. 集成到神经网络中的测试")
    print("=" * 50)


    class TestNet(nn.Module):
        def __init__(self):
            super(TestNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.se = SE_Block(64)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.se(x)
            x = self.conv2(x)
            return x


    test_net = TestNet()
    test_input = torch.randn(2, 3, 32, 32)
    test_output = test_net(test_input)

    print(f"网络输入维度: {test_input.shape}")
    print(f"网络输出维度: {test_output.shape}")

    # 7. 性能测试
    print("\n" + "=" * 50)
    print("7. 性能测试")
    print("=" * 50)

    import time

    se_block.eval()
    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = se_block(x)

        # 计时
        num_iterations = 100
        start_time = time.time()

        for _ in range(num_iterations):
            _ = se_block(x)

        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations * 1000  # 毫秒

        print(f"平均前向传播时间: {avg_time:.3f} ms")
        print(f"FPS: {1000 / avg_time:.1f}")

    print("\n" + "=" * 50)
    print("所有测试完成！")
    print("=" * 50)
