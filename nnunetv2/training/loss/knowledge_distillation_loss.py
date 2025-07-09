#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识蒸馏复合损失函数
- 封装了分割损失 (Dice + CE) 和两个分支间的知识蒸馏损失 (KL散度)。
"""
from torch import nn
import torch
import torch.nn.functional as F
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss

class DC_and_CE_and_KD_loss(nn.Module):
    def __init__(self,
                 seg_loss_kwargs: dict,
                 kd_weight: float = 0.1,
                 temperature: float = 2.0):
        """
        Args:
            seg_loss_kwargs (dict): 传递给内部 DC_and_CE_loss 的参数字典。
            kd_weight (float): 知识蒸馏损失在总损失中的权重。
            temperature (float): 蒸馏温度，用于平滑概率分布。
        """
        super().__init__()
        # 内部实例化一个标准的分割损失函数
        self.seg_loss = DC_and_CE_loss(**seg_loss_kwargs)
        
        # 知识蒸馏损失 (KL散度)
        self.kd_loss_fn = nn.KLDivLoss(log_target=True, reduction='batchmean')
        self.kd_weight = kd_weight
        self.temperature = temperature

    def forward(self,
                final_output: torch.Tensor,
                independent_logits_long: torch.Tensor,
                independent_logits_short: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        计算总损失。
        Args:
            final_output (torch.Tensor): 融合后的最终输出。
            independent_logits_long (torch.Tensor): 长轴分支的独立输出。
            independent_logits_short (torch.Tensor): 短轴分支的独立输出。
            target (torch.Tensor): 真实标签。
        """
        # 1. 计算分割损失
        # 我们只对最终的融合输出计算分割损失，因为这是模型的最终答案
        loss_seg = self.seg_loss(final_output, target)

        # 2. 计算知识蒸馏损失
        # 让两个独立分支互相学习，作为一种正则化
        kd_loss = self.kd_loss_fn(
            F.log_softmax(independent_logits_long / self.temperature, dim=1),
            F.log_softmax(independent_logits_short / self.temperature, dim=1)
        )

        # 3. 计算加权后的总损失
        total_loss = loss_seg + self.kd_weight * kd_loss
        
        return total_loss