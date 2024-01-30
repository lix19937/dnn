
import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Bottom-up pathway
        c1, c2, c3, c4 = x  # x 是输入   
        
        # Top-down pathway
        p4 = self.conv4(c4)  # 得到一张特征图
        p3 = self.conv3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest') # 这里将conv3(c3)得第二张特征图与第一张特征图进行融合，由于尺寸不同要进行上采样
        p2 = self.conv2(c2) + F.interpolate(p3, scale_factor=2, mode='nearest') # 同理得到三张特征图的融合图像
        p1 = self.conv1(c1) + F.interpolate(p2, scale_factor=2, mode='nearest') # 同理得四张特征图的融合图像


# Example usage
in_channels = 64    # 输入特征图的通道数
out_channels = 256  # 输出特征图的通道数

# 创建一个FPN模型
fpn = FPN(in_channels, out_channels) # 创建了一个FPN模型，输入为64通道，输出为256通道    

# 假设输入特征图的尺寸为(1, 64, 128, 128)，即(batch_size, in_channels, height, width)
input_features = torch.randn(1, in_channels, 128, 128)#随机生成输入

# 前向传播得到特征金字塔的各层特征
p1, p2, p3, p4 = fpn(input_features)#前向传播

print(p1.shape)  # 输出特征图的尺寸为(1, out_channels, 128, 128)
print(p2.shape)  # 输出特征图的尺寸为(1, out_channels, 64, 64)
print(p3.shape)  # 输出特征图的尺寸为(1, out_channels, 32, 32)
print(p4.shape)  # 输出特征图的尺寸为(1, out_channels, 16, 16)
