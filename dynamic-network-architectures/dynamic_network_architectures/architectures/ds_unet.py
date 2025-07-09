from .unet import PlainConvUNet
from ..building_blocks.depthwise_separable_conv import DepthwiseSeparableConvBlock

from dynamic_network_architectures.initialization.depthwise_init import DSConvInitWeights

class DS_PlainConvUNet(PlainConvUNet):
    def __init__(self, *args, **kwargs):
        # 使用类引用而非lambda表达式
        kwargs['conv_op'] = DepthwiseSeparableConvBlock

        print(f"[Network Architecture] 当前卷积模块类型: {kwargs['conv_op'].__name__}")  # 验证卷积类型

        
        # 显式定义二维卷积参数
        kwargs['kernel_sizes'] = [[3, 3]] * 8  # 8个阶段的卷积核
        kwargs['strides'] = [[1, 1]] + [[2, 2]] * 7  # 首层stride=1，后续7层stride=2
        super().__init__(*args, **kwargs)

    @staticmethod
    def initialize(module):
        # 使用深度可分离卷积专用初始化
        DSConvInitWeights(1e-2)(module)