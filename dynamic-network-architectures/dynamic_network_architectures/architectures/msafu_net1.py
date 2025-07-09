#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双分支MSAFUNet网络

支持：
- 两个独立分支处理长轴和短轴输入。
- 通过注意力机制融合多尺度特征。
- 返回分割输出和融合特征，用于一致性损失。

作者：多尺度融合团队
日期：2025年7月

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import  Tuple
class DualBranchMSAFUNet(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, deep_supervision: bool = False, 
                 n_stages: int = 4, features_per_stage: list = [32, 64, 128, 256], 
                 kernel_sizes: list = [3, 3, 3, 3], strides: list = [1, 2, 2, 2]):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        # 长轴和短轴分支（共享参数）
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels if i == 0 else features_per_stage[i-1], features_per_stage[i], 
                          kernel_size=kernel_sizes[i], stride=strides[i], padding=1, bias=True),
                nn.InstanceNorm2d(features_per_stage[i]),
                nn.LeakyReLU(inplace=True)
            ) for i in range(n_stages)
        ])
        
        # 解码器（简单上采样）
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(features_per_stage[i], features_per_stage[i-1], 
                                  kernel_size=2, stride=2, padding=0),
                nn.InstanceNorm2d(features_per_stage[i-1]),
                nn.LeakyReLU(inplace=True)
            ) for i in range(n_stages-1, 0, -1)
        ])
        
        # 分割头
        self.seg_head = nn.Conv2d(features_per_stage[0], num_classes, kernel_size=1)
        
        # 注意力融合模块
        self.attention = nn.Sequential(
            nn.Conv2d(features_per_stage[-1] * 2, features_per_stage[-1], kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x_long: torch.Tensor, x_short: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 长轴分支编码
        long_features = []
        x = x_long
        for enc in self.encoder:
            x = enc(x)
            long_features.append(x)
        
        # 短轴分支编码
        short_features = []
        x = x_short
        for enc in self.encoder:
            x = enc(x)
            short_features.append(x)
        
        # 特征融合（最后一层特征）
        fused_features = self.attention(torch.cat([long_features[-1], short_features[-1]], dim=1))
        fused_long = long_features[-1] * fused_features
        fused_short = short_features[-1] * fused_features
        
        # 长轴解码
        x = fused_long
        for dec in self.decoder:
            x = dec(x)
        output_long = self.seg_head(x)
        
        # 短轴解码
        x = fused_short
        for dec in self.decoder:
            x = dec(x)
        output_short = self.seg_head(x)
        
        return output_long, output_short, fused_long, fused_short
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 双分支多尺度注意力融合U-Net（DualBranchMSAFUNet）

# 该模块实现了一个双分支U-Net架构，用于处理颈动脉斑块超声图像的长轴和短轴。
# 每个分支是一个MSAFUNet，包含多尺度注意力模块，捕获不同尺度的特征。
# 通过一致性损失关联两个分支，鼓励学习相似的语义特征。

# 作者：多尺度融合团队
# 日期：2025年7月
# """

# from typing import Union, Type, List, Tuple
# import torch
# from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
# from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
# from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
# from dynamic_network_architectures.initialization.weight_init import InitWeights_He
# from dynamic_network_architectures.architectures.unet import PlainConvUNet
# from torch import nn
# from torch.nn.modules.conv import _ConvNd
# from torch.nn.modules.dropout import _DropoutNd

# class MultiScaleAttention(nn.Module):
#     """
#     多尺度注意力模块。

#     通过并行卷积核（1x1、3x3、5x5）捕获多尺度特征，融合后通过注意力机制加权。
#     """
#     def __init__(self, in_channels: int, out_channels: int):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
#         self.fuse = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
#         self.attention = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels // 8, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels // 8, out_channels, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x1 = self.conv1(x)
#         x3 = self.conv3(x)
#         x5 = self.conv5(x)
#         fused = self.fuse(torch.cat([x1, x3, x5], dim=1))
#         attention_map = self.attention(fused)
#         return fused * attention_map

# class MSAFUNet(nn.Module):
#     """
#     多尺度注意力融合U-Net。

#     包含标准U-Net编码器、多尺度注意力模块和解码器，支持深度监督。
#     """
#     def __init__(self,
#                  input_channels: int,
#                  n_stages: int,
#                  features_per_stage: Union[int, List[int], Tuple[int, ...]],
#                  conv_op: Type[_ConvNd],
#                  kernel_sizes: Union[int, List[int], Tuple[int, ...]],
#                  strides: Union[int, List[int], Tuple[int, ...]],
#                  n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
#                  num_classes: int,
#                  n_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]],
#                  conv_bias: bool = False,
#                  norm_op: Union[None, Type[nn.Module]] = None,
#                  norm_op_kwargs: dict = None,
#                  dropout_op: Union[None, Type[_DropoutNd]] = None,
#                  dropout_op_kwargs: dict = None,
#                  nonlin: Union[None, Type[nn.Module]] = None,
#                  nonlin_kwargs: dict = None,
#                  deep_supervision: bool = False,
#                  nonlin_first: bool = False):
#         super().__init__()
#         if isinstance(n_conv_per_stage, int):
#             n_conv_per_stage = [n_conv_per_stage] * n_stages
#         if isinstance(n_conv_per_stage_decoder, int):
#             n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
#         assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage长度必须等于阶段数"
#         assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder长度必须比阶段数少1"

#         self.encoder = PlainConvEncoder(
#             input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
#             n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
#             dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
#             nonlin_first=nonlin_first
#         )
#         self.msa_modules = nn.ModuleList([
#             MultiScaleAttention(features_per_stage[i], features_per_stage[i]) 
#             for i in range(n_stages - 1)
#         ])
#         self.decoder = UNetDecoder(
#             self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
#             nonlin_first=nonlin_first
#         )

#     def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
#         skips = self.encoder(x)
#         processed_skips = [self.msa_modules[i](skip) for i, skip in enumerate(skips[:-1])]
#         processed_skips.append(skips[-1])
#         return self.decoder(processed_skips)

# class DualBranchMSAFUNet(nn.Module):
#     """
#     双分支MSAFUNet。

#     包含两个MSAFUNet实例，分别处理长轴和短轴图像，通过一致性损失关联。
#     """
#     def __init__(self, input_channels: int, n_stages: int, features_per_stage: list,
#                  conv_op: type, kernel_sizes: list, strides: list, n_conv_per_stage: list,
#                  num_classes: int, n_conv_per_stage_decoder: list, conv_bias: bool = True,
#                  norm_op: type = None, norm_op_kwargs: dict = None, dropout_op: type = None,
#                  dropout_op_kwargs: dict = None, nonlin: type = None, nonlin_kwargs: dict = None,
#                  deep_supervision: bool = True):
#         super().__init__()
#         self.net_long_axis = PlainConvUNet(
#             input_channels=input_channels,
#             n_stages=n_stages,
#             features_per_stage=features_per_stage,
#             conv_op=conv_op,
#             kernel_sizes=kernel_sizes,
#             strides=strides,
#             n_conv_per_stage=n_conv_per_stage,
#             num_classes=num_classes,
#             n_conv_per_stage_decoder=n_conv_per_stage_decoder,
#             conv_bias=conv_bias,
#             norm_op=norm_op,
#             norm_op_kwargs=norm_op_kwargs,
#             dropout_op=dropout_op,
#             dropout_op_kwargs=dropout_op_kwargs,
#             nonlin=nonlin,
#             nonlin_kwargs=nonlin_kwargs,
#             deep_supervision=deep_supervision
#         )
#         self.net_short_axis = PlainConvUNet(
#             input_channels=input_channels,
#             n_stages=n_stages,
#             features_per_stage=features_per_stage,
#             conv_op=conv_op,
#             kernel_sizes=kernel_sizes,
#             strides=strides,
#             n_conv_per_stage=n_conv_per_stage,
#             num_classes=num_classes,
#             n_conv_per_stage_decoder=n_conv_per_stage_decoder,
#             conv_bias=conv_bias,
#             norm_op=norm_op,
#             norm_op_kwargs=norm_op_kwargs,
#             dropout_op=dropout_op,
#             dropout_op_kwargs=dropout_op_kwargs,
#             nonlin=nonlin,
#             nonlin_kwargs=nonlin_kwargs,
#             deep_supervision=deep_supervision
#         )
#         self.decoder = self.net_long_axis.decoder  # 添加decoder属性以兼容nnUNet
#         self.apply(InitWeights_He(1e-2))

#     def forward(self, x_long: torch.Tensor, x_short: torch.Tensor = None, return_features: bool = False):
#         """
#         前向传播。

#         Args:
#             x_long (torch.Tensor): 长轴输入图像，形状 [batch_size, input_channels, H, W]
#             x_short (torch.Tensor, optional): 短轴输入图像，形状 [batch_size, input_channels, H, W]。若为None，则使用x_long。
#             return_features (bool): 是否返回特征图，用于一致性损失。

#         Returns:
#             tuple: (output_long, output_short, features_long, features_short)
#                 - output_long: 长轴分支的预测，list（深度监督）或torch.Tensor
#                 - output_short: 短轴分支的预测，list（深度监督）或torch.Tensor
#                 - features_long: 长轴分支的特征图（最后一层，torch.Tensor）
#                 - features_short: 短轴分支的特征图（最后一层，torch.Tensor）
#         """
#         if x_short is None:
#             x_short = x_long  # 兼容nnUNet的plot_network_architecture

#         # 长轴分支前向传播
#         output_long = self.net_long_axis(x_long)
#         # 提取特征图（最后一层解码器输出或主输出）
#         features_long = output_long[-1] if isinstance(output_long, (list, tuple)) else output_long

#         # 短轴分支前向传播
#         output_short = self.net_short_axis(x_short)
#         # 提取特征图
#         features_short = output_short[-1] if isinstance(output_short, (list, tuple)) else output_short

#         return output_long, output_short, features_long, features_short

#     def compute_conv_feature_map_size(self, input_size: Tuple[int, ...]) -> int:
#         """计算卷积特征图大小，适配nnUNet规划。"""
#         assert len(input_size) == convert_conv_op_to_dim(self.net_long_axis.encoder.conv_op), "输入尺寸应为图像尺寸"
#         size_long = self.net_long_axis.compute_conv_feature_map_size(input_size)
#         size_short = self.net_short_axis.compute_conv_feature_map_size(input_size)
#         return size_long + size_short