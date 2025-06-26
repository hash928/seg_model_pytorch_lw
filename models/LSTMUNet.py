import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,  # 对应i, f, g, o四个门
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        
        # 分割门控信号
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)
        
        # 应用门控
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        # 更新细胞状态
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class LSTMUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, hidden_channels=64):
        super(LSTMUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.hidden_channels = hidden_channels

        # 编码器
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels*2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(inplace=True)
        )
        
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels*2, hidden_channels*4, 3, padding=1),
            nn.BatchNorm2d(hidden_channels*4),
            nn.ReLU(inplace=True)
        )
        
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels*4, hidden_channels*8, 3, padding=1),
            nn.BatchNorm2d(hidden_channels*8),
            nn.ReLU(inplace=True)
        )
        
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels*8, hidden_channels*16, 3, padding=1),
            nn.BatchNorm2d(hidden_channels*16),
            nn.ReLU(inplace=True)
        )

        # LSTM层
        self.lstm1 = ConvLSTMCell(hidden_channels*2, hidden_channels*2, 3)
        self.lstm2 = ConvLSTMCell(hidden_channels*4, hidden_channels*4, 3)
        self.lstm3 = ConvLSTMCell(hidden_channels*8, hidden_channels*8, 3)
        self.lstm4 = ConvLSTMCell(hidden_channels*16, hidden_channels*16, 3)

        # 解码器
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels*16, hidden_channels*8, 2, stride=2),
            nn.BatchNorm2d(hidden_channels*8),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels*8, hidden_channels*4, 2, stride=2),
            nn.BatchNorm2d(hidden_channels*4),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels*4, hidden_channels*2, 2, stride=2),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels*2, hidden_channels, 2, stride=2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        self.outc = nn.Conv2d(hidden_channels, n_classes, 1)

    def forward(self, x):
        # 编码器路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 初始化LSTM状态
        batch_size, _, height, width = x2.size()
        h1 = torch.zeros(batch_size, self.hidden_channels*2, height, width).to(x.device)
        c1 = torch.zeros(batch_size, self.hidden_channels*2, height, width).to(x.device)
        
        height, width = x3.size()[2:]
        h2 = torch.zeros(batch_size, self.hidden_channels*4, height, width).to(x.device)
        c2 = torch.zeros(batch_size, self.hidden_channels*4, height, width).to(x.device)
        
        height, width = x4.size()[2:]
        h3 = torch.zeros(batch_size, self.hidden_channels*8, height, width).to(x.device)
        c3 = torch.zeros(batch_size, self.hidden_channels*8, height, width).to(x.device)
        
        height, width = x5.size()[2:]
        h4 = torch.zeros(batch_size, self.hidden_channels*16, height, width).to(x.device)
        c4 = torch.zeros(batch_size, self.hidden_channels*16, height, width).to(x.device)

        # LSTM处理
        h1, c1 = self.lstm1(x2, h1, c1)
        h2, c2 = self.lstm2(x3, h2, c2)
        h3, c3 = self.lstm3(x4, h3, c3)
        h4, c4 = self.lstm4(x5, h4, c4)

        # 解码器路径
        x = self.up1(h4)
        x = self.up2(x + h3)
        x = self.up3(x + h2)
        x = self.up4(x + h1)
        
        logits = self.outc(x)
        return logits