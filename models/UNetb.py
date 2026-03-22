import sys
import os
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from module.ASPP import ASPP
from module.Attention import SE_Block, CBAM_Block, CA_Block, ECA_Block


feat_maps = {}
def save_feature(name):
    """特征热图提取"""
    def hook(module, input, output):
        feat_maps[name] = output.detach().cpu()
    return hook

def unet_conv(in_channels, out_channels, padding=1):
    """
    UNet网络中block的基本实现，encode和decode类似，都是2个3x3卷积。
    与论文中的实现不同的是，默认加了padding都是same卷积
    :param in_channels: 输入channels
    :param out_channels: 输出channels
    :param padding: 默认加padding
    :return:
    """
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=padding, bias=False),
                         # 第一个3x3卷积，后面接bn，bias=False
                         nn.BatchNorm2d(out_channels),  # bn
                         nn.ReLU(inplace=True),  # relu激活函数
                         nn.Conv2d(out_channels, out_channels, 3, padding=padding, bias=False),
                         # 第二个3x3卷积，后面接bn，bias=False
                         nn.BatchNorm2d(out_channels),  # bn
                         nn.ReLU(inplace=True))  # relu激活函数


class _UNetEncoder(nn.Module):
    def __init__(self, encode_blocks):
        """
        encoder部分。
        第一个encode block没有下采样。
        最后一个encode block没有shortcut。
        :param encode_blocks: encode每个层的block
        """
        super(_UNetEncoder, self).__init__()
        self.encoder = nn.ModuleList(encode_blocks)  # module保存所有的encode block
        pass

    def forward(self, x):
        shortcuts = []  # 保存所有的shortcut
        for encode in self.encoder:
            x = encode(x)  # 每个encode block的输出都保存
            shortcuts.append(x)
        return x, shortcuts[:-1]  # 返回提取的特征x，和所有shortcut。最后一个block输出不作为shortcut。

    pass


class _UNetDecoder(nn.Module):
    def __init__(self, encode_out_channels, n_class, use_aspp=False, attention_type=None):
        """
        decoder部分。
        decode block比encode block少一个。
        所有的decode block都是上采样up -> 和shortcut拼接 -> decode操作。
        decode操作都是类似的unet_conv，是两个3x3卷积。与论文实现不同的是默认加padding使用same卷积。
        最后，增加一个1x1卷积，用于最后的分类。
        :param encode_out_channels: 列表类型，每个encode block的输出channels，按照encode block的顺序。
        :param n_class: n种分类。
        :param use_aspp: 是否使用ASPP模块。
        :param attention_type: 注意力模块类型，可选: "se", "cbam", "ca", "eca" 或 None。
        """
        super(_UNetDecoder, self).__init__()
        self.use_aspp = use_aspp
        self.attention_type = attention_type
        self.ups = nn.ModuleList()  # 上采样
        self.decodes = nn.ModuleList()  # decode

        in_channels = encode_out_channels[-1]  # 最后一个encode block的输出channels作为decode的输入channels
        
        # 如果使用ASPP，在bottleneck处添加ASPP模块
        if use_aspp:
            self.aspp = ASPP(in_channels, [6, 12, 18], 256)
            in_channels = 256  # ASPP输出256个通道
        
        # 根据attention_type添加相应的注意力模块
        if attention_type == "se":
            self.attention = SE_Block(in_channels, ratio=16)
        elif attention_type == "cbam":
            self.attention = CBAM_Block(in_channels, reduction=16, spatial_kernel=7)
        elif attention_type == "ca":
            # CA模块需要输入和输出通道数相同
            self.attention = CA_Block(in_channels, in_channels, groups=32)
        elif attention_type == "eca":
            self.attention = ECA_Block(in_channels, gamma=2, b=1)
        else:
            self.attention = None
        
        for cat_channels in reversed(encode_out_channels[:-1]):  # decode与encode顺序相反，遍历所有剩余的encode block的输出channels
            out_channels = in_channels // 2  # 上采样输出channels是输入channels的一半,spatial增大一倍
            self.ups.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            )

            in_channels = out_channels + cat_channels  # 与shortcut进行cat，改变了decode的输入channels
            # out_channels = in_channels // 2  # decode输出channels是输入channels的一半
            self.decodes.append(unet_conv(in_channels, out_channels))  # decode，decode是类似的都是2个3x3的same卷积

            in_channels = out_channels  # decode的输入channels作为下一次迭代的输入channels
            pass
        self.classifier = nn.Conv2d(in_channels, n_class, 1)  # 1x1卷积得到最终分类预测
        pass

    def forward(self, x, shortcuts):
        # 如果使用ASPP，在bottleneck处应用ASPP
        if self.use_aspp:
            x = self.aspp(x)
        
        # 如果使用注意力模块，在bottleneck处应用
        if self.attention is not None:
            x = self.attention(x)
            
        for i, (up, decode) in enumerate(zip(self.ups, self.decodes)):
            x = up(x)  # 先上采样
            x, s = self._crop(x, shortcuts[-i - 1])  # 剪裁大小，因为下采样上采样等会使x和shortcut的spatial大小不一致
            x = torch.cat([x, s], dim=1)  # 沿dim=1，也就是channel方向cat
            x = decode(x)  # decode，decode是类似的都是2个3x3的same卷积
        x = self.classifier(x)  # 1x1卷积得到最终分类预测
        return x

    @staticmethod
    def _crop(x, shortcut):
        """
        按照x和shortcut最小值剪裁
        :param x: 上采样结果
        :param shortcut: 就是shortcut
        :return: 剪裁后的x和shortcut
        """
        _, _, h_x, w_x = x.shape  # 取特征的spatial大小
        _, _, h_s, w_s = shortcut.shape  # 取shortcut的spatial大小
        h, w = min(h_x, h_s), min(w_x, w_s)  # 取最小spatial
        hc_x, wc_x = (h_x - h) // 2, (w_x - w) // 2  # x要剪裁掉的值
        hc_s, wc_s = (h_s - h) // 2, (w_s - w) // 2  # shortcut要剪裁掉的值
        x = x[..., hc_x:hc_x + h, wc_x: wc_x + w]  # center crop
        shortcut = shortcut[..., hc_s:hc_s + h, wc_s:wc_s + w]  # center crop
        return x, shortcut

    pass


class _UNetFactory(nn.Module):
    def __init__(self, encode_blocks, encode_out_channels, n_class,
                 init_encoder=True, init_decoder=True, use_aspp=False, attention_type=None):
        """
        UNet工厂类，用于生成UNet模型的网络。
        :param encode_blocks: 列表类型。列表每个元素是一个encode的block
        :param encode_out_channels: 列表类型。列表每个元素是encode block的输出channels，按照encode的顺序。
        :param n_class: n种分类。
        :param init_encoder: 是否初始化encoder的权重。ResNet修改了encoder部分，一般不需要初始化。
        :param init_decoder: 是否初始化decoder的权重。decoder一般一样，都需要初始化。
        :param use_aspp: 是否使用ASPP模块。
        :param attention_type: 注意力模块类型，可选: "se", "cbam", "ca", "eca" 或 None。
        """
        super(_UNetFactory, self).__init__()
        self.encoder = _UNetEncoder(encode_blocks)
        self.decoder = _UNetDecoder(encode_out_channels, n_class, use_aspp, attention_type)

        # 初始化参数
        if init_encoder:
            for m in self.encoder.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        if init_decoder:
            for m in self.decoder.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        pass

    def forward(self, x):
        x, shortcuts = self.encoder(x)
        x = self.decoder(x, shortcuts)
        return x

    pass


def unet_base(in_channels, n_class, use_aspp=False, attention_type=None):
    """
    按照论文实现的unet网络，与论文不同的是使用了same卷积。
    :param in_channels: 输入channels，也就是image的channels
    :param n_class: n种分类
    :param use_aspp: 是否使用ASPP模块
    :param attention_type: 注意力模块类型，可选: "se", "cbam", "ca", "eca" 或 None
    :return: unet网络
    """
    encode_blocks = [unet_conv(in_channels, 64)]
    for i in range(4):
        encode_blocks.append(nn.Sequential(nn.MaxPool2d(2, stride=2, ceil_mode=True),
                                           unet_conv(64 * 2 ** i, 128 * 2 ** i)))
    encode_out_channels = [64, 128, 256, 512, 1024]
    return _UNetFactory(encode_blocks, encode_out_channels, n_class, use_aspp=use_aspp, attention_type=attention_type)


def unet_resnet(resnet_type, in_channels, n_class, pretrained=True, use_aspp=False, attention_type=None):
    """
    用resnet预训练模型作为encoder实现的unet网络。
    :param resnet_type: resnet类型。可以是resnet18/34/50/101/152
    :param in_channels: 输入channels，也就是image的channels
    :param n_class: n种分类
    :param pretrained: 是否使用预训练权重
    :param use_aspp: 是否使用ASPP模块
    :param attention_type: 注意力模块类型，可选: "se", "cbam", "ca", "eca" 或 None
    :return: 使用resnet作为backbone的unet网络
    """
    if resnet_type == 'resnet18':
        resnet = resnet18(pretrained=pretrained)
        encode_out_channels = [in_channels, 64, 64, 128, 256, 512]
    elif resnet_type == 'resnet34':
        resnet = resnet34(pretrained=pretrained)
        encode_out_channels = [in_channels, 64, 64, 128, 256, 512]
    elif resnet_type == 'resnet50':
        resnet = resnet50(pretrained=pretrained)
        encode_out_channels = [in_channels, 64, 256, 512, 1024, 2048]
    elif resnet_type == 'resnet101':
        resnet = resnet101(pretrained=pretrained)
        encode_out_channels = [in_channels, 64, 256, 512, 1024, 2048]
    elif resnet_type == 'resnet152':
        resnet = resnet152(pretrained=pretrained)
        encode_out_channels = [in_channels, 64, 256, 512, 1024, 2048]
    else:
        raise ValueError('resnet type error!')
    encode_blocks = [nn.Sequential(),  # 1x，第1个encode block什么都不做
                     nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu),  # 2x，resnet的conv1_x进行第1次下采样
                     nn.Sequential(resnet.maxpool, resnet.layer1),  # 4x，resnet的maxpool进行第2次下采样，conv2_x不进行下采样
                     resnet.layer2,  # 8x，resnet的conv3_x进行第3次下采样
                     resnet.layer3,  # 16x，resnet的conv4_x进行第4次下采样
                     resnet.layer4]  # 32x，resnet的conv5_x进行第5次下采样
    return _UNetFactory(encode_blocks, encode_out_channels, n_class,
                        init_encoder=not pretrained, use_aspp=use_aspp, attention_type=attention_type)  # 有pretrain的encoder不初始化


if __name__ == '__main__':
    dev = torch.device('cuda:0')
    model = unet_resnet('resnet34', 3, 2, use_aspp=True, attention_type="se")  # resnet18作为backbone的unet
    # 注册到 decoder 的 attention
    model.decoder.attention.register_forward_hook(
        save_feature("attention_bottleneck")
    )
    # 解码器钩子
    model.decoder.decodes[1].register_forward_hook(
        save_feature("decoder_stage1")
    )
    # 编码器钩子
    model.encoder.encoder[4].register_forward_hook(
        save_feature("encoder_layer4")
    )
    model.to(dev)  # 装入gpu
    model.eval()
    print(model)  # 打印看模型是否正确



    in_data = torch.randint(0, 256, (1, 3, 352, 352), dtype=torch.float)  # 测试输入
    in_data = in_data.to(dev)  # 装入gpu
    print(in_data.shape)

    # 先进行前向传播，hook才会被触发并保存特征图
    out_data = model(in_data)
    print(out_data.shape)  # 输出应该是1x2x572x572的tensor

    # 前向传播后，特征图已经被保存，可以访问
    if "attention_bottleneck" in feat_maps:
        feat = feat_maps["attention_bottleneck"]
        print(f"特征图形状: {feat.shape}")
    else:
        print("警告: 未找到特征图，请检查hook是否正确注册")
    pass


    from PIL import Image
    import torchvision.transforms as T

    img_path = "/home/data/sam-unet/shi_ce/xiangdao_data12/Training_Images/yuanxing_jz_label_54.png"
    img = Image.open(img_path).convert("RGB")

    transform = T.Compose([
        T.ToTensor(),  # [0,1]
        # T.Resize((352, 352)),
    ])

    img_tensor = transform(img).unsqueeze(0).to(dev)

    feat_maps.clear()

    # 加载权重并处理键名映射和类别数不匹配问题
    checkpoint_path = "/home/seg_model1/checkpoints/unet_resnet34_aspp_se_dice_bce_loss_training_Mosaic_4_seed512/unet_resnet34-200.pth"
    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)

    # 处理键名映射：将 decoder.se.* 映射到 decoder.attention.*
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 创建新的状态字典，处理键名映射
    new_state_dict = {}
    for key, value in state_dict.items():
        # 将 decoder.se.* 映射到 decoder.attention.*
        if key.startswith('decoder.se.'):
            new_key = key.replace('decoder.se.', 'decoder.attention.')
            new_state_dict[new_key] = value
        # 跳过分类器层（因为类别数不匹配）
        elif 'decoder.classifier' in key:
            print(f"跳过分类器层: {key} (形状不匹配: 检查点 {value.shape} vs 模型 {model.state_dict()[key].shape})")
            continue
        else:
            new_state_dict[key] = value

    # 部分加载权重（strict=False 允许跳过不匹配的键）
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        print(f"警告: 以下键未加载: {missing_keys}")
    if unexpected_keys:
        print(f"警告: 以下键未使用: {unexpected_keys}")
    print("权重加载完成（已跳过分类器层）")

    model.eval()

    with torch.no_grad():
        _ = model(img_tensor)

    feat = feat_maps["attention_bottleneck"]

    heatmap = feat.squeeze(0).mean(dim=0).numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    import matplotlib.pyplot as plt

    # 特征图
    plt.figure(figsize=(10, 3), facecolor='none', frameon=False)
    plt.imshow(heatmap, cmap="jet", aspect="auto")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 调整子图边距
    plt.show()


    plt.figure(figsize=(10, 3), facecolor='none', frameon=False)
    plt.imshow(img, aspect='auto')  # 关键参数：aspect='auto'
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 完全去掉边距
    plt.show()
