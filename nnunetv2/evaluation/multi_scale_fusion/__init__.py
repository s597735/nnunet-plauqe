# -*- coding: utf-8 -*-
"""
多尺度特征融合评估模块

本模块包含多尺度特征融合的评估实现，包括：
- 多尺度融合评估器
- 性能指标计算
- 结果可视化

Author: Multi-Scale Fusion Team
Date: 2024
"""

from .evaluate_multi_scale_fusion import (
    MultiScaleFusionEvaluator,
    evaluate_multi_scale_predictions,
    compute_advanced_metrics
)

__all__ = [
    'MultiScaleFusionEvaluator',
    'evaluate_multi_scale_predictions',
    'compute_advanced_metrics'
]