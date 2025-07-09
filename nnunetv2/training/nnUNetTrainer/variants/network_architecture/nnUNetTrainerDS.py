from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from dynamic_network_architectures.architectures.ds_unet import DS_PlainConvUNet
import torch.nn as nn

class nnUNetTrainerDS(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: list,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        if arch_init_kwargs.get('use_ds_conv', False):
            return DS_PlainConvUNet(
                input_channels=num_input_channels,
                num_classes=num_output_channels,
                n_stages=arch_init_kwargs['n_stages'],
                features_per_stage=arch_init_kwargs['features_per_stage'],
                conv_op=arch_init_kwargs['conv_op'],
                kernel_sizes=arch_init_kwargs['kernel_sizes'],
                strides=arch_init_kwargs['strides'],
                deep_supervision=enable_deep_supervision
            )
        return super().build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision
        )