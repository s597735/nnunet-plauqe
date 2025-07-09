from typing import Union, List, Tuple  # 新增此行
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
import numpy as np

class DCAExperimentPlanner(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor',
                 plans_name: str = 'nnUNetDCAPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        # 初始化DCA参数
        self.use_dca = True
        self.dca_in_channels = 32

    def get_plans_for_configuration(self, spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache):
        # 调用父类生成基础配置
        plans = super().get_plans_for_configuration(spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache)
        
        # 确保configurations字段存在
        if 'configurations' not in plans:
            plans['configurations'] = {}
        
        # 解析config_name（例如："Dataset1001_3d_fullres" -> "3d_fullres"）
        config_name = data_identifier.split('_')[-1]
        
        # 初始化当前配置的dca_params
        if config_name not in plans['configurations']:
            plans['configurations'][config_name] = {}
        if 'dca_params' not in plans['configurations'][config_name]:
            plans['configurations'][config_name]['dca_params'] = {}
        
        # 注入DCA参数
        plans['configurations'][config_name]['dca_params'].update({
            'use_dca': self.use_dca,
            'dca_in_channels': self.dca_in_channels
        })
        return plans