from typing import Union, Tuple, Type  # 添加Type类型声明
import torch
from torch import nn

class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Union[int, Tuple[int, ...]],
                 padding: Union[int, Tuple[int, ...]],
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 groups: int = 1,  # 必须保留但实际会被覆盖
                 padding_mode: str = 'zeros',
                 # 参数名必须与PyTorch标准卷积层对齐 ▼▼▼
                 bias: bool = False,  # 原conv_bias改为标准名称bias
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[nn.Module]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False):
        super().__init__()
        
        # 显式覆盖groups参数确保深度卷积
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # 关键参数不可修改
            bias=bias,
            padding_mode=padding_mode
        )
        
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias  # 保持相同参数名
        )
        
        # 标准化层
        self.norm = norm_op(out_channels, **norm_op_kwargs) if norm_op else nn.Identity()
        
        # 非线性激活
        self.nonlin = nonlin(**nonlin_kwargs) if nonlin else nn.Identity()
        
        # 正则化
        self.dropout = dropout_op(**dropout_op_kwargs) if dropout_op else nn.Identity()
        
        self.nonlin_first = nonlin_first

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        
        if self.nonlin_first:
            x = self.nonlin(x)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.nonlin(x)
            
        return self.dropout(x)