#!/usr/bin/env python3
"""多尺度特征融合模型评估脚本"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 添加nnUNet路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from batchgenerators.utilities.file_and_folder_operations import join, load_json, save_json, isfile, isdir, maybe_mkdir_p, subfiles
import SimpleITK as sitk


class MultiScaleFusionEvaluator:
    """多尺度特征融合模型评估器"""
    
    def __init__(self, reference_folder: str, prediction_folder: str, 
                 output_folder: str, labels: Optional[Dict[int, str]] = None):
        
        self.reference_folder = reference_folder
        self.prediction_folder = prediction_folder
        self.output_folder = output_folder
        self.labels = labels or {0: 'Background', 1: 'Plaque', 2: 'Vessel', 3: 'Calcification', 4: 'Lipid'}
        
        # 创建输出文件夹
        maybe_mkdir_p(self.output_folder)
        
        # 初始化结果存储
        self.results = {}
        self.case_results = []
        
    def evaluate_all_cases(self, metrics: List[str] = None) -> Dict:
        """评估所有case"""
        
        if metrics is None:
            metrics = ['Dice', 'HD95', 'ASSD', 'Precision', 'Recall', 'Specificity']
        
        print(f"Evaluating predictions in: {self.prediction_folder}")
        print(f"Against references in: {self.reference_folder}")
        print(f"Metrics to compute: {metrics}")
        
        # 获取所有预测文件
        pred_files = subfiles(self.prediction_folder, suffix='.nii.gz', join=False)
        
        if not pred_files:
            raise ValueError(f"No prediction files found in {self.prediction_folder}")
        
        print(f"Found {len(pred_files)} prediction files")
        
        # 评估每个case
        for pred_file in pred_files:
            case_id = pred_file.replace('.nii.gz', '')
            self._evaluate_single_case(case_id, metrics)
        
        # 计算总体统计
        self._compute_summary_statistics()
        
        # 保存结果
        self._save_results()
        
        # 生成可视化
        self._generate_visualizations()
        
        return self.results
    
    def _evaluate_single_case(self, case_id: str, metrics: List[str]):
        """评估单个case"""
        
        pred_file = join(self.prediction_folder, f"{case_id}.nii.gz")
        ref_file = join(self.reference_folder, f"{case_id}.nii.gz")
        
        if not isfile(ref_file):
            print(f"Warning: Reference file not found for {case_id}")
            return
        
        if not isfile(pred_file):
            print(f"Warning: Prediction file not found for {case_id}")
            return
        
        try:
            # 加载图像
            pred_img = sitk.ReadImage(pred_file)
            ref_img = sitk.ReadImage(ref_file)
            
            pred_array = sitk.GetArrayFromImage(pred_img)
            ref_array = sitk.GetArrayFromImage(ref_img)
            
            # 确保数据类型一致
            pred_array = pred_array.astype(np.int32)
            ref_array = ref_array.astype(np.int32)
            
            # 计算指标
            case_metrics = self._compute_case_metrics(pred_array, ref_array, metrics)
            case_metrics['case_id'] = case_id
            
            self.case_results.append(case_metrics)
            
            print(f"Evaluated {case_id}: Dice = {case_metrics.get('Dice_mean', 'N/A'):.4f}")
            
        except Exception as e:
            print(f"Error evaluating {case_id}: {e}")
    
    def _compute_case_metrics(self, pred: np.ndarray, ref: np.ndarray, metrics: List[str]) -> Dict:
        """计算单个case的指标"""
        
        case_metrics = {}
        unique_labels = np.unique(ref)
        unique_labels = unique_labels[unique_labels > 0]  # 排除背景
        
        # 为每个标签计算指标
        label_metrics = {}
        
        for label in unique_labels:
            pred_mask = (pred == label)
            ref_mask = (ref == label)
            
            label_results = {}
            
            if 'Dice' in metrics:
                dice = self._compute_dice(pred_mask, ref_mask)
                label_results['Dice'] = dice
            
            if 'HD95' in metrics:
                hd95 = self._compute_hausdorff_95(pred_mask, ref_mask)
                label_results['HD95'] = hd95
            
            if 'ASSD' in metrics:
                assd = self._compute_assd(pred_mask, ref_mask)
                label_results['ASSD'] = assd
            
            if 'Precision' in metrics:
                precision = self._compute_precision(pred_mask, ref_mask)
                label_results['Precision'] = precision
            
            if 'Recall' in metrics:
                recall = self._compute_recall(pred_mask, ref_mask)
                label_results['Recall'] = recall
            
            if 'Specificity' in metrics:
                specificity = self._compute_specificity(pred_mask, ref_mask)
                label_results['Specificity'] = specificity
            
            label_metrics[f'Label_{label}'] = label_results
        
        # 计算平均指标
        for metric in metrics:
            values = [label_metrics[label_key][metric] for label_key in label_metrics.keys() 
                     if metric in label_metrics[label_key] and not np.isnan(label_metrics[label_key][metric])]
            
            if values:
                case_metrics[f'{metric}_mean'] = np.mean(values)
                case_metrics[f'{metric}_std'] = np.std(values)
            else:
                case_metrics[f'{metric}_mean'] = np.nan
                case_metrics[f'{metric}_std'] = np.nan
        
        # 添加标签特定指标
        case_metrics.update(label_metrics)
        
        return case_metrics
    
    def _compute_dice(self, pred: np.ndarray, ref: np.ndarray) -> float:
        """计算Dice系数"""
        intersection = np.logical_and(pred, ref).sum()
        total = pred.sum() + ref.sum()
        
        if total == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return 2.0 * intersection / total
    
    def _compute_hausdorff_95(self, pred: np.ndarray, ref: np.ndarray) -> float:
        """计算95%Hausdorff距离"""
        try:
            from scipy.spatial.distance import directed_hausdorff
            from scipy.ndimage import binary_erosion
            
            # 获取边界点
            pred_boundary = pred ^ binary_erosion(pred)
            ref_boundary = ref ^ binary_erosion(ref)
            
            pred_points = np.argwhere(pred_boundary)
            ref_points = np.argwhere(ref_boundary)
            
            if len(pred_points) == 0 or len(ref_points) == 0:
                return np.inf
            
            # 计算双向Hausdorff距离
            dist1 = directed_hausdorff(pred_points, ref_points)[0]
            dist2 = directed_hausdorff(ref_points, pred_points)[0]
            
            # 返回95百分位数
            distances = np.concatenate([
                np.min(np.linalg.norm(pred_points[:, None] - ref_points, axis=2), axis=1),
                np.min(np.linalg.norm(ref_points[:, None] - pred_points, axis=2), axis=1)
            ])
            
            return np.percentile(distances, 95)
            
        except Exception:
            return np.nan
    
    def _compute_assd(self, pred: np.ndarray, ref: np.ndarray) -> float:
        """计算平均对称表面距离"""
        try:
            from scipy.ndimage import binary_erosion
            
            # 获取边界点
            pred_boundary = pred ^ binary_erosion(pred)
            ref_boundary = ref ^ binary_erosion(ref)
            
            pred_points = np.argwhere(pred_boundary)
            ref_points = np.argwhere(ref_boundary)
            
            if len(pred_points) == 0 or len(ref_points) == 0:
                return np.inf
            
            # 计算最近邻距离
            pred_to_ref = np.min(np.linalg.norm(pred_points[:, None] - ref_points, axis=2), axis=1)
            ref_to_pred = np.min(np.linalg.norm(ref_points[:, None] - pred_points, axis=2), axis=1)
            
            # 计算平均对称表面距离
            assd = (np.mean(pred_to_ref) + np.mean(ref_to_pred)) / 2
            
            return assd
            
        except Exception:
            return np.nan
    
    def _compute_precision(self, pred: np.ndarray, ref: np.ndarray) -> float:
        """计算精确率"""
        tp = np.logical_and(pred, ref).sum()
        fp = np.logical_and(pred, ~ref).sum()
        
        if tp + fp == 0:
            return 1.0 if tp == 0 else 0.0
        
        return tp / (tp + fp)
    
    def _compute_recall(self, pred: np.ndarray, ref: np.ndarray) -> float:
        """计算召回率"""
        tp = np.logical_and(pred, ref).sum()
        fn = np.logical_and(~pred, ref).sum()
        
        if tp + fn == 0:
            return 1.0 if tp == 0 else 0.0
        
        return tp / (tp + fn)
    
    def _compute_specificity(self, pred: np.ndarray, ref: np.ndarray) -> float:
        """计算特异性"""
        tn = np.logical_and(~pred, ~ref).sum()
        fp = np.logical_and(pred, ~ref).sum()
        
        if tn + fp == 0:
            return 1.0 if tn == 0 else 0.0
        
        return tn / (tn + fp)
    
    def _compute_summary_statistics(self):
        """计算总体统计"""
        
        if not self.case_results:
            print("No case results to summarize")
            return
        
        df = pd.DataFrame(self.case_results)
        
        # 计算总体统计
        summary_stats = {}
        
        metric_columns = [col for col in df.columns if col.endswith('_mean')]
        
        for col in metric_columns:
            metric_name = col.replace('_mean', '')
            values = df[col].dropna()
            
            if len(values) > 0:
                summary_stats[metric_name] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'median': float(values.median()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'q25': float(values.quantile(0.25)),
                    'q75': float(values.quantile(0.75)),
                    'count': int(len(values))
                }
        
        self.results['summary_statistics'] = summary_stats
        self.results['case_results'] = self.case_results
        self.results['num_cases'] = len(self.case_results)
    
    def _save_results(self):
        """保存结果"""
        
        # 保存JSON格式结果
        json_file = join(self.output_folder, 'evaluation_results.json')
        save_json(self.results, json_file)
        
        # 保存CSV格式的case结果
        if self.case_results:
            df = pd.DataFrame(self.case_results)
            csv_file = join(self.output_folder, 'case_results.csv')
            df.to_csv(csv_file, index=False)
            
            # 保存汇总统计
            if 'summary_statistics' in self.results:
                summary_df = pd.DataFrame(self.results['summary_statistics']).T
                summary_csv = join(self.output_folder, 'summary_statistics.csv')
                summary_df.to_csv(summary_csv)
        
        print(f"Results saved to: {self.output_folder}")
    
    def _generate_visualizations(self):
        """生成可视化图表"""
        
        if not self.case_results:
            return
        
        df = pd.DataFrame(self.case_results)
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Dice系数分布
        if 'Dice_mean' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 2, 1)
            sns.histplot(df['Dice_mean'].dropna(), bins=20, kde=True)
            plt.title('Dice Coefficient Distribution')
            plt.xlabel('Dice Coefficient')
            plt.ylabel('Frequency')
        
        # 2. 多指标箱线图
        metric_columns = [col for col in df.columns if col.endswith('_mean') and col.replace('_mean', '') in ['Dice', 'Precision', 'Recall']]
        if metric_columns:
            plt.subplot(2, 2, 2)
            metric_data = []
            metric_names = []
            for col in metric_columns:
                values = df[col].dropna()
                if len(values) > 0:
                    metric_data.append(values)
                    metric_names.append(col.replace('_mean', ''))
            
            if metric_data:
                plt.boxplot(metric_data, labels=metric_names)
                plt.title('Metric Distributions')
                plt.ylabel('Score')
                plt.xticks(rotation=45)
        
        # 3. 指标相关性热图
        if len(metric_columns) > 1:
            plt.subplot(2, 2, 3)
            corr_data = df[metric_columns].corr()
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                       xticklabels=[col.replace('_mean', '') for col in metric_columns],
                       yticklabels=[col.replace('_mean', '') for col in metric_columns])
            plt.title('Metric Correlations')
        
        # 4. Case性能排序
        if 'Dice_mean' in df.columns:
            plt.subplot(2, 2, 4)
            sorted_df = df.sort_values('Dice_mean', ascending=False)
            plt.plot(range(len(sorted_df)), sorted_df['Dice_mean'], 'o-', markersize=3)
            plt.title('Case Performance (sorted by Dice)')
            plt.xlabel('Case Rank')
            plt.ylabel('Dice Coefficient')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(join(self.output_folder, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成详细的标签特定分析
        self._generate_label_specific_plots(df)
    
    def _generate_label_specific_plots(self, df: pd.DataFrame):
        """生成标签特定的分析图表"""
        
        # 提取标签特定的指标
        label_columns = [col for col in df.columns if col.startswith('Label_')]
        
        if not label_columns:
            return
        
        # 为每个标签创建图表
        unique_labels = set()
        for col in label_columns:
            label_num = col.split('_')[1]
            unique_labels.add(int(label_num))
        
        if len(unique_labels) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            metrics = ['Dice', 'Precision', 'Recall', 'HD95']
            
            for i, metric in enumerate(metrics):
                if i >= len(axes):
                    break
                
                ax = axes[i]
                label_data = []
                label_names = []
                
                for label_num in sorted(unique_labels):
                    # 尝试从case_results中提取标签特定数据
                    values = []
                    for case in self.case_results:
                        label_key = f'Label_{label_num}'
                        if label_key in case and metric in case[label_key]:
                            val = case[label_key][metric]
                            if not np.isnan(val):
                                values.append(val)
                    
                    if values:
                        label_data.append(values)
                        label_name = self.labels.get(label_num, f'Label {label_num}')
                        label_names.append(label_name)
                
                if label_data:
                    ax.boxplot(label_data, labels=label_names)
                    ax.set_title(f'{metric} by Label')
                    ax.set_ylabel(metric)
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(join(self.output_folder, 'label_specific_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def compare_with_baseline(self, baseline_folder: str, output_file: str = None):
        """与基线模型比较"""
        
        print(f"Comparing with baseline results in: {baseline_folder}")
        
        # 评估基线模型
        baseline_evaluator = MultiScaleFusionEvaluator(
            self.reference_folder, baseline_folder, 
            join(self.output_folder, 'baseline_eval'), self.labels
        )
        baseline_results = baseline_evaluator.evaluate_all_cases()
        
        # 比较结果
        comparison = self._perform_statistical_comparison(baseline_results)
        
        # 保存比较结果
        if output_file is None:
            output_file = join(self.output_folder, 'baseline_comparison.json')
        
        save_json(comparison, output_file)
        
        # 生成比较可视化
        self._generate_comparison_plots(baseline_results, comparison)
        
        return comparison
    
    def _perform_statistical_comparison(self, baseline_results: Dict) -> Dict:
        """执行统计比较"""
        
        comparison = {
            'multi_scale_fusion': self.results['summary_statistics'],
            'baseline': baseline_results['summary_statistics'],
            'statistical_tests': {},
            'improvements': {}
        }
        
        # 对每个指标进行统计检验
        for metric in self.results['summary_statistics'].keys():
            if metric in baseline_results['summary_statistics']:
                
                # 提取数据
                msf_values = [case[f'{metric}_mean'] for case in self.case_results 
                             if f'{metric}_mean' in case and not np.isnan(case[f'{metric}_mean'])]
                baseline_values = [case[f'{metric}_mean'] for case in baseline_results['case_results'] 
                                  if f'{metric}_mean' in case and not np.isnan(case[f'{metric}_mean'])]
                
                if len(msf_values) > 0 and len(baseline_values) > 0:
                    # 执行配对t检验
                    if len(msf_values) == len(baseline_values):
                        t_stat, p_value = stats.ttest_rel(msf_values, baseline_values)
                        test_type = 'paired_t_test'
                    else:
                        t_stat, p_value = stats.ttest_ind(msf_values, baseline_values)
                        test_type = 'independent_t_test'
                    
                    # 计算效应大小 (Cohen's d)
                    pooled_std = np.sqrt(((len(msf_values) - 1) * np.var(msf_values, ddof=1) + 
                                         (len(baseline_values) - 1) * np.var(baseline_values, ddof=1)) / 
                                        (len(msf_values) + len(baseline_values) - 2))
                    
                    cohens_d = (np.mean(msf_values) - np.mean(baseline_values)) / pooled_std
                    
                    comparison['statistical_tests'][metric] = {
                        'test_type': test_type,
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'cohens_d': float(cohens_d),
                        'significant': p_value < 0.05
                    }
                    
                    # 计算改进百分比
                    msf_mean = np.mean(msf_values)
                    baseline_mean = np.mean(baseline_values)
                    improvement_pct = ((msf_mean - baseline_mean) / baseline_mean) * 100
                    
                    comparison['improvements'][metric] = {
                        'absolute_improvement': float(msf_mean - baseline_mean),
                        'relative_improvement_pct': float(improvement_pct),
                        'msf_mean': float(msf_mean),
                        'baseline_mean': float(baseline_mean)
                    }
        
        return comparison
    
    def _generate_comparison_plots(self, baseline_results: Dict, comparison: Dict):
        """生成比较图表"""
        
        # 创建比较图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 指标改进条形图
        ax1 = axes[0, 0]
        metrics = list(comparison['improvements'].keys())
        improvements = [comparison['improvements'][m]['relative_improvement_pct'] for m in metrics]
        
        bars = ax1.bar(metrics, improvements)
        ax1.set_title('Relative Improvement over Baseline (%)')
        ax1.set_ylabel('Improvement (%)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 为正改进着色为绿色，负改进着色为红色
        for bar, imp in zip(bars, improvements):
            bar.set_color('green' if imp > 0 else 'red')
            bar.set_alpha(0.7)
        
        # 2. 指标对比散点图
        ax2 = axes[0, 1]
        msf_means = [comparison['improvements'][m]['msf_mean'] for m in metrics]
        baseline_means = [comparison['improvements'][m]['baseline_mean'] for m in metrics]
        
        ax2.scatter(baseline_means, msf_means, s=100, alpha=0.7)
        
        # 添加对角线
        min_val = min(min(msf_means), min(baseline_means))
        max_val = max(max(msf_means), max(baseline_means))
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax2.set_xlabel('Baseline Performance')
        ax2.set_ylabel('Multi-Scale Fusion Performance')
        ax2.set_title('Performance Comparison')
        
        # 添加标签
        for i, metric in enumerate(metrics):
            ax2.annotate(metric, (baseline_means[i], msf_means[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 3. p值可视化
        ax3 = axes[1, 0]
        p_values = [comparison['statistical_tests'][m]['p_value'] for m in metrics 
                   if m in comparison['statistical_tests']]
        test_metrics = [m for m in metrics if m in comparison['statistical_tests']]
        
        bars = ax3.bar(test_metrics, [-np.log10(p) for p in p_values])
        ax3.set_title('Statistical Significance (-log10(p-value))')
        ax3.set_ylabel('-log10(p-value)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
        ax3.legend()
        
        # 为显著结果着色
        for bar, p in zip(bars, p_values):
            bar.set_color('green' if p < 0.05 else 'gray')
            bar.set_alpha(0.7)
        
        # 4. 效应大小
        ax4 = axes[1, 1]
        effect_sizes = [comparison['statistical_tests'][m]['cohens_d'] for m in test_metrics]
        
        bars = ax4.bar(test_metrics, effect_sizes)
        ax4.set_title("Effect Size (Cohen's d)")
        ax4.set_ylabel("Cohen's d")
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Small effect')
        ax4.axhline(y=0.5, color='blue', linestyle='--', alpha=0.5, label='Medium effect')
        ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large effect')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(join(self.output_folder, 'baseline_comparison_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Evaluate Multi-Scale Fusion nnUNet predictions')
    
    parser.add_argument('-ref', '--reference', type=str, required=True,
                       help='Folder containing reference segmentations')
    parser.add_argument('-pred', '--prediction', type=str, required=True,
                       help='Folder containing prediction segmentations')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Output folder for evaluation results')
    parser.add_argument('--baseline', type=str, default=None,
                       help='Folder containing baseline predictions for comparison')
    parser.add_argument('--labels', type=str, default=None,
                       help='JSON file containing label definitions')
    parser.add_argument('--metrics', type=str, nargs='+', 
                       default=['Dice', 'HD95', 'ASSD', 'Precision', 'Recall'],
                       help='Metrics to compute')
    
    args = parser.parse_args()
    
    # 加载标签定义
    labels = None
    if args.labels and isfile(args.labels):
        labels = load_json(args.labels)
        # 转换键为整数
        labels = {int(k): v for k, v in labels.items()}
    
    print(f"Evaluation configuration:")
    print(f"  Reference: {args.reference}")
    print(f"  Prediction: {args.prediction}")
    print(f"  Output: {args.output}")
    print(f"  Baseline: {args.baseline}")
    print(f"  Metrics: {args.metrics}")
    print(f"  Labels: {labels}")
    
    try:
        # 创建评估器
        evaluator = MultiScaleFusionEvaluator(
            reference_folder=args.reference,
            prediction_folder=args.prediction,
            output_folder=args.output,
            labels=labels
        )
        
        # 执行评估
        results = evaluator.evaluate_all_cases(metrics=args.metrics)
        
        # 打印汇总结果
        print("\nEvaluation Results:")
        print("=" * 50)
        
        if 'summary_statistics' in results:
            for metric, stats in results['summary_statistics'].items():
                print(f"{metric}:")
                print(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"  Median [Q25, Q75]: {stats['median']:.4f} [{stats['q25']:.4f}, {stats['q75']:.4f}]")
                print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print()
        
        # 与基线比较
        if args.baseline:
            print("Comparing with baseline...")
            comparison = evaluator.compare_with_baseline(args.baseline)
            
            print("\nComparison with Baseline:")
            print("=" * 50)
            
            for metric, improvement in comparison['improvements'].items():
                print(f"{metric}:")
                print(f"  Improvement: {improvement['relative_improvement_pct']:.2f}%")
                print(f"  MSF: {improvement['msf_mean']:.4f}, Baseline: {improvement['baseline_mean']:.4f}")
                
                if metric in comparison['statistical_tests']:
                    test = comparison['statistical_tests'][metric]
                    significance = "***" if test['p_value'] < 0.001 else "**" if test['p_value'] < 0.01 else "*" if test['p_value'] < 0.05 else "ns"
                    print(f"  Statistical significance: {significance} (p={test['p_value']:.4f})")
                print()
        
        print(f"Evaluation completed! Results saved to: {args.output}")
    
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()