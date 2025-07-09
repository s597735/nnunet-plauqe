import torch
import torch.nn as nn

class DualCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//8, in_channels, 1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, skip):
        # x: 解码器特征, skip: 编码器特征
        channel_weight = self.channel_att(skip)
        channel_out = channel_weight * x
        
        spatial_weight = self.spatial_att(skip)
        return spatial_weight * channel_out + x  # 残差连接