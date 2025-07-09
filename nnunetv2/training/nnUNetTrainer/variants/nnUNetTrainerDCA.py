import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.dual_cross_attention import DualCrossAttention
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
class DCAWrapper(nn.Module):
    def __init__(self, dca_module, stage_module):
        super().__init__()
        self.dca = dca_module
        self.stage = stage_module
        
    def forward(self, x):
        # 在这里，x 是一个包含 skip connection 的 tuple
        if isinstance(x, (tuple, list)):
            skip = x[1] if len(x) > 1 else None
            x = x[0]
            if skip is not None:
                x = self.dca(x, skip)
        x = self.stage(x)
        return x

class nnUNetTrainerDCA(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
    
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                               arch_init_kwargs: dict,
                               arch_init_kwargs_req_import: list,
                               num_input_channels: int,
                               num_output_channels: int,
                               enable_deep_supervision: bool = True) -> nn.Module:
        # 首先调用父类方法创建基础网络
        network = super(nnUNetTrainerDCA, nnUNetTrainerDCA).build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision
        )
        
        # 应用DCA修改
        return nnUNetTrainerDCA.modify_network_architecture(network)
        
    @staticmethod
    def modify_network_architecture(network: nn.Module, logger=None) -> nn.Module:
        """
        静态方法用于修改网络结构，添加DCA模块
        Args:
            network: 要修改的网络
            logger: 可选的日志记录器
        Returns:
            修改后的网络
        """
        def log_message(msg):
            if logger:
                logger(msg)
            print(msg)
            
        try:
            decoder = getattr(network, "decoder", None)
            if decoder is not None and hasattr(decoder, "stages"):
                if len(decoder.stages) > 0:
                    dca_in_channels = 32  # 设置输入通道数
                    old_stage = decoder.stages[0]
                    # 使用包装器来处理skip connection
                    decoder.stages[0] = DCAWrapper(
                        DualCrossAttention(dca_in_channels),
                        old_stage
                    )
                    log_message("[DCA] DualCrossAttention已成功插入decoder第0阶段")
                    log_message(f"[DCA] 修改后的stage[0]结构: {decoder.stages[0]}")
                else:
                    log_message("[DCA] decoder.stages为空，DCA插入失败")
            else:
                log_message("[DCA] 未找到decoder或decoder.stages，DCA插入失败")
        except Exception as e:
            log_message(f"[DCA] 插入DualCrossAttention时发生异常: {e}")
            raise e
            
        return network
    
    def initialize(self):
        super().initialize()
        # 不需要再次修改网络结构，因为在build_network_architecture中已经完成了
        self.print_to_log_file("==== Network Structure (nnUNetTrainerDCA) ====")
        self.print_to_log_file(str(self.network))
        self.print_to_log_file("==========================================")