import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticConsistencyLoss(nn.Module):
    """
    语义一致性损失函数。

    该损失函数旨在惩罚两个不同网络（或同一网络的不同分支）输出之间的差异，
    常用于处理非配对的多模态或多视图数据，例如本项目的长轴和短轴超声图像。

    它假设即使输入来自不同视图，但对于同一个解剖结构，网络的输出在语义层面应该是一致的。
    损失计算通常基于网络输出的概率图（经过Softmax或Sigmoid激活后）。
    """
    def __init__(self, loss_type='mse'):
        """
        初始化语义一致性损失。

        参数:
            loss_type (str): 用于计算一致性损失的类型。可选值为 'mse' (均方误差) 或 'kl' (KL散度)。
                             默认为 'mse'。
        """
        super(SemanticConsistencyLoss, self).__init__()
        if loss_type not in ['mse', 'kl']:
            raise ValueError(f"不支持的损失类型: {loss_type}。请选择 'mse' 或 'kl'。")
        self.loss_type = loss_type

    def forward(self, pred1: torch.Tensor, pred2: torch.Tensor) -> torch.Tensor:
        """
        计算前向传播过程中的一致性损失。

        参数:
            pred1 (torch.Tensor): 第一个网络分支的输出概率图。形状为 (b, c, x, y, z)。
            pred2 (torch.Tensor): 第二个网络分支的输出概率图。形状为 (b, c, x, y, z)。

        返回:
            torch.Tensor: 计算出的一致性损失值。
        """
        # 确保输入是经过softmax的概率分布
        p1 = F.softmax(pred1, dim=1)
        p2 = F.softmax(pred2, dim=1)

        if self.loss_type == 'mse':
            # 计算均方误差损失
            consistency_loss = F.mse_loss(p1, p2)
        elif self.loss_type == 'kl':
            # 计算对称KL散度损失
            kl_div1 = F.kl_div(p1.log(), p2, reduction='batchmean')
            kl_div2 = F.kl_div(p2.log(), p1, reduction='batchmean')
            consistency_loss = (kl_div1 + kl_div2) / 2.0
        
        return consistency_loss