from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from dynamic_network_architectures.architectures.msafu_net import DualBranchMSAFUNet

class MSAFUNetPlanner(ExperimentPlanner):
    """
    MSAFUNet 实验计划器。

    该计划器继承自 nnU-Net 的默认实验计划器（ExperimentPlanner），
    其主要作用是为 MSAFUNet 模型生成训练计划（plans file）。
    它会自动配置网络参数，并指定使用 'MSAFUNet' 作为网络架构。
    """
    def __init__(self, dataset_name_or_id: str, gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', 
                 plans_name: str = 'MSAFUNetPlans', # 指定生成的计划文件名称
                 overwrite_target_spacing: list = None, 
                 suppress_transpose: bool = False):
        """初始化函数"""
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name, 
                         overwrite_target_spacing, suppress_transpose)
                         
        self.UNet_class = DualBranchMSAFUNet
