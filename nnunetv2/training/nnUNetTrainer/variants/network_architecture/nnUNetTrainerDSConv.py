from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from dynamic_network_architectures.architectures.ds_unet import DS_PlainConvUNet  # 修正导入路径
from dynamic_network_architectures.building_blocks.depthwise_separable_conv import DSConvInitWeights

class nnUNetTrainerDSConv(nnUNetTrainer):
    def configure_architecture(self):
        return DS_PlainConvUNet(**self._get_network_init_params())  # 添加网络配置方法
    
    @staticmethod
    def initialize(module):
        DSConvInitWeights(1e-2)(module)  # 使用专用初始化