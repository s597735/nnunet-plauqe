#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mean Teacher 一致性损失函数模块
- 封装了分割损失 (Dice + CE) 和教师-学生模型间的一致性损失。
"""
from torch import nn
import torch
import torch.nn.functional as F
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss

class DC_and_CE_and_Consistency_loss(nn.Module):
    def __init__(self,
                 seg_loss_kwargs: dict,
                 consistency_weight: float = 0.1,
                 consistency_loss_fn: nn.Module = nn.MSELoss()):
        """
        Args:
            seg_loss_kwargs (dict): 传递给内部 DC_and_CE_loss 的参数字典。
            consistency_weight (float): 一致性损失在总损失中的权重。
            consistency_loss_fn (nn.Module): 用于计算一致性损失的函数，默认为MSELoss。
        """
        super().__init__()
        # 内部实例化一个标准的分割损失函数
        self.seg_loss = DC_and_CE_loss(**seg_loss_kwargs)
        
        # 保存一致性损失相关的参数
        self.consistency_weight = consistency_weight
        self.consistency_loss_fn = consistency_loss_fn

    def forward(self, student_output: torch.Tensor, target: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
        """
        计算总损失。
        Args:
            student_output (torch.Tensor): 学生模型的输出。
            target (torch.Tensor): 真实标签。
            teacher_output (torch.Tensor): 教师模型的输出。
        """
        # 1. 计算分割损失
        # 我们只对学生模型的输出计算分割损失，因为只有学生模型需要学习去拟合真实标签
        loss_seg = self.seg_loss(student_output, target)

        # 2. 计算一致性损失
        # 教师模型的输出是稳定的监督信号，学生模型需要向它看齐
        # 我们对softmax后的概率图计算损失，这比KL散度更稳定
        loss_consistency = self.consistency_loss_fn(
            F.softmax(student_output, dim=1),
            F.softmax(teacher_output, dim=1)
        )

        # 3. 计算加权后的总损失
        total_loss = loss_seg + self.consistency_weight * loss_consistency
        
        return total_loss