import torch
import torch.nn as nn
import torch.nn.functional as F

class StripAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(StripAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        # 水平和垂直条带注意力
        self.h_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), groups=in_channels)
        self.v_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0), groups=in_channels)
        
        # 通道降维和升维
        self.channel_reduction = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.channel_expansion = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 水平条带注意力
        h_att = self.h_conv(x)
        
        # 垂直条带注意力
        v_att = self.v_conv(x)
        
        # 融合两种条带注意力
        fused_att = h_att + v_att
        fused_att = self.act(fused_att)
        
        # 通道注意力
        fused_att = self.channel_reduction(fused_att)
        fused_att = self.act(fused_att)
        fused_att = self.channel_expansion(fused_att)
        
        # 生成注意力权重
        att_weights = self.sigmoid(fused_att)
        
        # 应用注意力
        return x * att_weights

class SCASeg(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(SCASeg, self).__init__()
        self.strip_attention = StripAttention(in_channels, reduction_ratio)
        
    def forward(self, x):
        return self.strip_attention(x) + x  # 残差连接