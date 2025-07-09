#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终决战版 - DualBranchMSAFUNet (瓶颈层融合 + EMA)
- 在网络的瓶颈层对两个分支的最高阶语义特征进行融合。
- 在融合前，使用EMA模块对瓶颈特征进行增强。
- 两个解码器都使用这个融合后的特征进行重建，实现真正的特征级信息共享。
- 此设计在保证稳定性的前提下，极大地提升了模型的性能潜力。
"""
from typing import Union, Type, List, Tuple
import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

class EMAModule(nn.Module):
    """ 高效多尺度注意力模块 (EMA) """
    def __init__(self, in_channels: int, conv_op: Type[nn.Module]):
        super().__init__()
        self.group_conv = conv_op(in_channels, in_channels, kernel_size=3,
        stride=1, padding=1, groups=in_channels, bias=False)
        self.cross_spatial = nn.Sequential(
            conv_op(in_channels, in_channels, kernel_size=1, bias=False),
            conv_op(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            conv_op(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x_g = self.group_conv(x)
        x_c = self.cross_spatial(x)
        attention_map = self.sigmoid(x_g + x_c)
        return identity * attention_map

class DualBranchMSAFUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: list,
                 conv_op: type,
                 kernel_sizes: list,
                 strides: list,
                 num_classes: int,
                 n_conv_per_stage: list,
                 n_conv_per_stage_decoder: list,
                 conv_bias: bool = True,
                 norm_op: type = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: type = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: type = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False):
        super().__init__()
        
        # 强制关闭两个内部网络的深度监督，确保行为绝对一致和稳定
        branch_kwargs = {
            'input_channels': input_channels, 'n_stages': n_stages, 'features_per_stage': features_per_stage,
            'conv_op': conv_op, 'kernel_sizes': kernel_sizes, 'strides': strides,
            'n_conv_per_stage': n_conv_per_stage, 'num_classes': num_classes,
            'n_conv_per_stage_decoder': n_conv_per_stage_decoder, 'conv_bias': conv_bias,
            'norm_op': norm_op, 'norm_op_kwargs': norm_op_kwargs, 'dropout_op': dropout_op,
            'dropout_op_kwargs': dropout_op_kwargs, 'nonlin': nonlin, 'nonlin_kwargs': nonlin_kwargs,
            'deep_supervision': False
        }
        
        self.net_long_axis = PlainConvUNet(**branch_kwargs)
        self.net_short_axis = PlainConvUNet(**branch_kwargs)
        
        # [!! 新增 1 !!] 瓶颈层的EMA注意力模块
        bottleneck_features = features_per_stage[-1]
        self.ema_long = EMAModule(bottleneck_features, conv_op)
        self.ema_short = EMAModule(bottleneck_features, conv_op)

        # [!! 新增 2 !!] 瓶颈层的特征融合卷积层
        self.bottleneck_fusion_conv = conv_op(bottleneck_features * 2, bottleneck_features, 
        kernel_size=1, stride=1, padding=0, bias=True)

        # 后期融合层依然保留
        self.fusion_conv = conv_op(num_classes * 2, num_classes, 
        kernel_size=1, stride=1, padding=0, bias=True)
        self.decoder = self.net_long_axis.decoder
        self.apply(InitWeights_He(1e-2))

    def forward(self, x_long: torch.Tensor, x_short: torch.Tensor = None) -> torch.Tensor:
        if x_short is None:
            x_short = x_long
        
        # 1. 两个分支的编码器独立运行，得到各自的跳跃连接和瓶颈特征
        skips_long = self.net_long_axis.encoder(x_long)
        skips_short = self.net_short_axis.encoder(x_short)
        
        bottleneck_long = skips_long[-1]
        bottleneck_short = skips_short[-1]

        # 2. 对瓶颈特征应用EMA模块进行增强
        ema_bottleneck_long = self.ema_long(bottleneck_long)
        ema_bottleneck_short = self.ema_short(bottleneck_short)

        # 3. 在瓶颈层进行特征融合
        concatenated_bottleneck = torch.cat((ema_bottleneck_long, ema_bottleneck_short), dim=1)
        fused_bottleneck = self.bottleneck_fusion_conv(concatenated_bottleneck)

        # 4. [!! 核心修正 !!]
        #    以 nnU-Net Decoder 期望的正确方式，将“原始跳跃连接”和“新的融合瓶颈”一同传入
        #    Decoder期望的输入是一个列表: [skip1, skip2, ..., skip_n, bottleneck]
        
        # 提取两个分支原始的、不包含瓶颈层的跳跃连接
        original_skips_long = skips_long[:-1]
        original_skips_short = skips_short[:-1]

        # 将原始跳跃连接和新的融合瓶颈层重新打包成解码器需要的格式
        decoder_input_long = original_skips_long + [fused_bottleneck]
        decoder_input_short = original_skips_short + [fused_bottleneck]

        # [!! 核心修正 !!]
        # 删除了错误的 "self.net_..._decoder.encoder.skips = ..." 注入方式
        # 使用正确的参数调用解码器
        logits_long = self.net_long_axis.decoder(decoder_input_long)
        logits_short = self.net_short_axis.decoder(decoder_input_short)
        
        # 5. 对两个更优的预测结果，进行最终的后期融合
        concatenated_logits = torch.cat((logits_long, logits_short), dim=1)
        final_output = self.fusion_conv(concatenated_logits)

        return final_output

    def compute_conv_feature_map_size(self, input_size: Tuple[int, ...]) -> int:
        return self.net_long_axis.compute_conv_feature_map_size(input_size)