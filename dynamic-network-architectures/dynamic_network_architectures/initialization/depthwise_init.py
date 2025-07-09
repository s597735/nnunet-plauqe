import torch.nn as nn
import math

class DSConvInitWeights:
    def __init__(self, slope=1e-2):
        self.slope = slope

    def __call__(self, module):
        if isinstance(module, nn.Conv2d):
            # 深度卷积初始化
            if module.groups == module.in_channels:
                nn.init.kaiming_normal_(module.weight, 
                    mode='fan_out',
                    nonlinearity='leaky_relu',
                    a=self.slope)
            # 逐点卷积初始化    
            elif module.kernel_size == (1, 1):
                nn.init.kaiming_normal_(module.weight, 
                    mode='fan_in',
                    nonlinearity='linear')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            # 普通卷积层初始化
            else:
                nn.init.kaiming_normal_(module.weight,
                    mode='fan_out',
                    nonlinearity='relu')
        # BatchNorm初始化
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)