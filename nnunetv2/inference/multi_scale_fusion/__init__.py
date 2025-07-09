# -*- coding: utf-8 -*-
"""
多尺度特征融合推理模块

本模块包含多尺度特征融合的推理实现，包括：
- 多尺度融合预测器
- 多视图推理策略
- 后处理方法

Author: Multi-Scale Fusion Team
Date: 2024
"""

from .predict_multi_scale_fusion import (
    MultiScaleFusionPredictor,
    predict_from_raw_data_multi_scale,
    predict_single_case_multi_scale
)

__all__ = [
    'MultiScaleFusionPredictor',
    'predict_from_raw_data_multi_scale',
    'predict_single_case_multi_scale'
]