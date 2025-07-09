#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终决战版 V2 - DualBranchMSAFUNet (渐进式融合 + 跨注意力)
- 包含两个独立的编码器和两个独立的解码器。
- 在解码器的每个尺度上，通过一个跨注意力模块进行特征对齐和融合。
- 实现了真正的、渐进式的、贯穿解码全程的特征交互。
"""
from typing import Union, Type, List, Tuple
import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

# [!! 最终关键修复 !!] 使用池化来降低计算量的轻量化跨注意力模块
class LightweightCrossAttentionFusion(nn.Module):
    def __init__(self, in_channels: int, conv_op: Type[nn.Module]):
        super().__init__()
        self.query_conv = conv_op(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = conv_op(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = conv_op(in_channels, in_channels, kernel_size=1)
        
        # 使用平均池化来大幅缩小 key 和 value 的空间尺寸
        self.pool = nn.AdaptiveAvgPool2d((32, 32)) # 将key/value固定池化到32x32

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_feat: torch.Tensor, key_value_feat: torch.Tensor) -> torch.Tensor:
        batch_size, C, height, width = query_feat.size()
        
        proj_query = self.query_conv(query_feat).view(batch_size, -1, width * height).permute(0, 2, 1)
        
        # 在计算 key 和 value 之前，先进行池化
        pooled_kv_feat = self.pool(key_value_feat)
        
        proj_key = self.key_conv(pooled_kv_feat).view(batch_size, -1, 32 * 32)
        proj_value = self.value_conv(pooled_kv_feat).view(batch_size, -1, 32 * 32)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)

        out = self.gamma * out + query_feat
        return out


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
        
        branch_kwargs = {
            'input_channels': input_channels, 'n_stages': n_stages, 'features_per_stage': features_per_stage,
            'conv_op': conv_op, 'kernel_sizes': kernel_sizes, 'strides': strides,
            'n_conv_per_stage': n_conv_per_stage, 'num_classes': num_classes,
            'n_conv_per_stage_decoder': n_conv_per_stage_decoder, 'conv_bias': conv_bias,
            'norm_op': norm_op, 'norm_op_kwargs': norm_op_kwargs, 'dropout_op': dropout_op,
            'dropout_op_kwargs': dropout_op_kwargs, 'nonlin': nonlin, 'nonlin_kwargs': nonlin_kwargs,
            'deep_supervision': False
        }
        
        # [!! 核心修改 1 !!] 我们不再实例化完整的U-Net，而是分别实例化编码器和解码器
        self.encoder_long = PlainConvUNet(**branch_kwargs).encoder
        self.encoder_short = PlainConvUNet(**branch_kwargs).encoder
        
        self.decoder_long = PlainConvUNet(**branch_kwargs).decoder
        self.decoder_short = PlainConvUNet(**branch_kwargs).decoder

        # [!! 核心修改 2 !!] 为解码器的每一层创建跨注意力融合模块
        self.fusion_modules_long_to_short = nn.ModuleList()
        self.fusion_modules_short_to_long = nn.ModuleList()
        # 解码器上采样后的特征通道数
        decoder_features = features_per_stage[::-1][1:] 
        for f in decoder_features:
            self.fusion_modules_long_to_short.append(LightweightCrossAttentionFusion(f, conv_op))
            self.fusion_modules_short_to_long.append(LightweightCrossAttentionFusion(f, conv_op))

        # 后期融合层依然保留
        self.fusion_conv = conv_op(num_classes * 2, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.apply(InitWeights_He(1e-2))

    def forward(self, x_long: torch.Tensor, x_short: torch.Tensor = None, return_independent_outputs: bool = False):
        if x_short is None: x_short = x_long
        
        # 1. 独立编码
        skips_long = self.encoder_long(x_long)
        skips_short = self.encoder_short(x_short)
        
        # 2. [!! 核心修改 3 !!] 渐进式跨注意力融合解码
        # 我们需要手动模拟解码器的上采样和融合过程
        
        # 初始化解码器的输入（来自bottleneck）
        x_dec_long = skips_long[-1]
        x_dec_short = skips_short[-1]

        for i in range(len(self.decoder_long.stages)):
            # a. 上采样
            x_dec_long = self.decoder_long.transpconvs[i](x_dec_long)
            x_dec_short = self.decoder_short.transpconvs[i](x_dec_short)

            # b. 跨注意力融合
            # 将长轴的上采样特征与短轴的跳跃连接进行注意力融合
            fused_for_short = self.fusion_modules_long_to_short[i](x_dec_short, skips_long[-(i + 2)])
            # 将短轴的上采样特征与长轴的跳跃连接进行注意力融合
            fused_for_long = self.fusion_modules_short_to_long[i](x_dec_long, skips_short[-(i + 2)])
            
            # c. 与各自的跳跃连接拼接
            x_dec_long = torch.cat((x_dec_long, fused_for_long), dim=1)
            x_dec_short = torch.cat((x_dec_short, fused_for_short), dim=1)
            
            # d. 通过解码器卷积块
            x_dec_long = self.decoder_long.stages[i](x_dec_long)
            x_dec_short = self.decoder_short.stages[i](x_dec_short)

        # 3. 最终输出和后期融合
        logits_long = self.decoder_long.seg_layers[0](x_dec_long)
        logits_short = self.decoder_short.seg_layers[0](x_dec_short)
        
        concatenated_logits = torch.cat((logits_long, logits_short), dim=1)
        final_output = self.fusion_conv(concatenated_logits)

        # 根据参数决定返回格式
        if return_independent_outputs:
            return final_output, logits_long, logits_short
        else:
            return final_output