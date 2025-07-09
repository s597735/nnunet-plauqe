from typing import Type
import numpy as np
import torch.nn
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.dropout import _DropoutNd
from torch.nn.modules.instancenorm import _InstanceNorm
import inspect


def convert_dim_to_conv_op(dimension: int) -> Type[_ConvNd]:
    """
    :param dimension: 1, 2 or 3
    :return: conv Class of corresponding dimension
    """
    if dimension == 1:
        return nn.Conv1d
    elif dimension == 2:
        return nn.Conv2d
    elif dimension == 3:
        return nn.Conv3d
    else:
        raise ValueError("Unknown dimension. Only 1, 2 and 3 are supported")


def convert_conv_op_to_dim(conv_op: Type[_ConvNd]) -> int:
    """
    :param conv_op: conv class
    :return: dimension: 1, 2 or 3
    """
    # 处理自定义深度可分离卷积模块
    if conv_op.__name__ == 'DepthwiseSeparableConvBlock':
        return 2
    # 处理标准PyTorch卷积类型
    if issubclass(conv_op, nn.Conv1d):
        return 1
    elif issubclass(conv_op, nn.Conv2d):
        return 2
    elif issubclass(conv_op, nn.Conv3d):
        return 3
    # 添加明确错误提示
    raise ValueError(f"Unsupported convolution type: {conv_op.__name__}. Supported types are: "
                     "DepthwiseSeparableConvBlock, Conv1d, Conv2d, Conv3d")



def get_matching_pool_op(conv_op: Type[_ConvNd] = None,
                         dimension: int = None,
                         adaptive=False,
                         pool_type: str = 'avg') -> Type[torch.nn.Module]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!
    :param conv_op:
    :param dimension:
    :param adaptive:
    :param pool_type: either 'avg' or 'max'
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    assert pool_type in ['avg', 'max'], 'pool_type must be either avg or max'
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'

    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)

    if dimension == 1:
        if pool_type == 'avg':
            if adaptive:
                return nn.AdaptiveAvgPool1d
            else:
                return nn.AvgPool1d
        elif pool_type == 'max':
            if adaptive:
                return nn.AdaptiveMaxPool1d
            else:
                return nn.MaxPool1d
    elif dimension == 2:
        if pool_type == 'avg':
            if adaptive:
                return nn.AdaptiveAvgPool2d
            else:
                return nn.AvgPool2d
        elif pool_type == 'max':
            if adaptive:
                return nn.AdaptiveMaxPool2d
            else:
                return nn.MaxPool2d
    elif dimension == 3:
        if pool_type == 'avg':
            if adaptive:
                return nn.AdaptiveAvgPool3d
            else:
                return nn.AvgPool3d
        elif pool_type == 'max':
            if adaptive:
                return nn.AdaptiveMaxPool3d
            else:
                return nn.MaxPool3d


def get_matching_instancenorm(conv_op: Type[_ConvNd] = None, dimension: int = None) -> Type[_InstanceNorm]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    if dimension is not None:
        assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'
    if dimension == 1:
        return nn.InstanceNorm1d
    elif dimension == 2:
        return nn.InstanceNorm2d
    elif dimension == 3:
        return nn.InstanceNorm3d


def get_matching_convtransp(conv_op: Type[_ConvNd] = None, dimension: int = None) -> Type[_ConvTransposeNd]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'
    if dimension == 1:
        return nn.ConvTranspose1d
    elif dimension == 2:
        return nn.ConvTranspose2d
    elif dimension == 3:
        return nn.ConvTranspose3d


def get_matching_batchnorm(conv_op: Type[_ConvNd] = None, dimension: int = None) -> Type[_BatchNorm]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'
    if dimension == 1:
        return nn.BatchNorm1d
    elif dimension == 2:
        return nn.BatchNorm2d
    elif dimension == 3:
        return nn.BatchNorm3d


def get_matching_dropout(conv_op: Type[_ConvNd] = None, dimension: int = None) -> Type[_DropoutNd]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'
    if dimension == 1:
        return nn.Dropout
    elif dimension == 2:
        return nn.Dropout2d
    elif dimension == 3:
        return nn.Dropout3d


def maybe_convert_scalar_to_list(conv_op, scalar):
    if isinstance(scalar, (list, tuple)):
        return list(scalar)
    # 统一使用convert_conv_op_to_dim处理维度
    dim = convert_conv_op_to_dim(conv_op)
    return [scalar] * dim

def get_default_network_config(dimension: int = 2,
                               nonlin: str = "ReLU",
                               norm_type: str = "bn") -> dict:
    """
    Use this to get a standard configuration. A network configuration looks like this:

    config = {'conv_op': torch.nn.modules.conv.Conv2d,
              'dropout_op': torch.nn.modules.dropout.Dropout2d,
              'norm_op': torch.nn.modules.batchnorm.BatchNorm2d,
              'norm_op_kwargs': {'eps': 1e-05, 'affine': True},
              'nonlin': torch.nn.modules.activation.ReLU,
              'nonlin_kwargs': {'inplace': True}}

    There is no need to use get_default_network_config. You can create your own. Network configs are a convenient way of
    setting dimensionality, normalization and nonlinearity.

    :param dimension: integer denoting the dimension of the data. 1, 2 and 3 are accepted
    :param nonlin: string (ReLU or LeakyReLU)
    :param norm_type: string (bn=batch norm, in=instance norm)
    torch.nn.Module
    :return: dict
    """
    config = {}
    config['conv_op'] = convert_dim_to_conv_op(dimension)
    config['dropout_op'] = get_matching_dropout(dimension=dimension)
    if norm_type == "bn":
        config['norm_op'] = get_matching_batchnorm(dimension=dimension)
    elif norm_type == "in":
        config['norm_op'] = get_matching_instancenorm(dimension=dimension)

    config['norm_op_kwargs'] = None # this will use defaults

    if nonlin == "LeakyReLU":
        config['nonlin'] = nn.LeakyReLU
        config['nonlin_kwargs'] = {'negative_slope': 1e-2, 'inplace': True}
    elif nonlin == "ReLU":
        config['nonlin'] = nn.ReLU
        config['nonlin_kwargs'] = {'inplace': True}
    else:
        raise NotImplementedError('Unknown nonlin %s. Only "LeakyReLU" and "ReLU" are supported for now' % nonlin)

    return config
