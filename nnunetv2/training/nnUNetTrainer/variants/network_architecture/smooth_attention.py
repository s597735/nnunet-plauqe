import torch
import torch.nn as nn
import torch.nn.functional as F

class SmoothAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(SmoothAttention, self).__init__()
        padding = kernel_size // 2
        
        # 平滑卷积
        self.smooth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                     padding=padding, groups=in_channels)
        
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 初始化为接近恒等映射
        nn.init.constant_(self.smooth_conv.weight, 0)
        nn.init.constant_(self.smooth_conv.bias, 0)
        for i in range(in_channels):
            self.smooth_conv.weight.data[i, i, padding, padding] = 1
        
    def forward(self, x):
        # 平滑特征
        smooth_x = self.smooth_conv(x)
        
        # 计算注意力权重
        att_weights = self.channel_att(smooth_x)
        
        # 应用注意力并添加残差连接
        return x + att_weights * smooth_x