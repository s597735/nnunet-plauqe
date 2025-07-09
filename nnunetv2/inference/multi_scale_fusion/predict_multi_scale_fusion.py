#!/usr/bin/env python3
"""多尺度特征融合模型推理脚本"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple

# 添加nnUNet路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.nnUNetTrainer.multi_scale_fusion.nnUNetTrainerMultiScaleFusionV2 import nnUNetTrainerMultiScaleFusionV2
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from batchgenerators.utilities.file_and_folder_operations import join, load_json, save_json, isfile, isdir, maybe_mkdir_p
import SimpleITK as sitk


class MultiScaleFusionPredictor(nnUNetPredictor):
    """多尺度特征融合模型预测器"""
    
    def __init__(self, tile_step_size: float = 0.5, use_gaussian: bool = True,
                 use_mirroring: bool = True, perform_everything_on_gpu: bool = True,
                 device: torch.device = torch.device('cuda'), verbose: bool = False,
                 verbose_preprocessing: bool = False, allow_tqdm: bool = True,
                 enable_multi_view: bool = False):
        
        super().__init__(tile_step_size, use_gaussian, use_mirroring, 
                        perform_everything_on_gpu, device, verbose, 
                        verbose_preprocessing, allow_tqdm)
        
        self.enable_multi_view = enable_multi_view
        self.multi_view_data = {}
    
    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                            use_folds: Union[Tuple[Union[int, str]], None] = None,
                                            checkpoint_name: str = 'checkpoint_final.pth'):
        """从训练好的模型文件夹初始化预测器"""
        
        # 调用父类方法
        super().initialize_from_trained_model_folder(model_training_output_dir, use_folds, checkpoint_name)
        
        # 检查是否为多尺度融合模型
        if hasattr(self.network, 'enable_multi_view'):
            self.enable_multi_view = self.network.enable_multi_view
            print(f"Multi-view mode: {self.enable_multi_view}")
    
    def predict_from_files(self, list_of_lists_or_source_folder: Union[str, List[List[str]]],
                          output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
                          save_probabilities: bool = False, overwrite: bool = True,
                          num_processes_preprocessing: int = 8, num_processes_segmentation_export: int = 8,
                          folder_with_segs_from_previous_stage: str = None, num_parts: int = 1, part_id: int = 0):
        """从文件预测"""
        
        if self.enable_multi_view:
            return self._predict_multi_view_from_files(
                list_of_lists_or_source_folder, output_folder_or_list_of_truncated_output_files,
                save_probabilities, overwrite, num_processes_preprocessing, 
                num_processes_segmentation_export, folder_with_segs_from_previous_stage,
                num_parts, part_id
            )
        else:
            return super().predict_from_files(
                list_of_lists_or_source_folder, output_folder_or_list_of_truncated_output_files,
                save_probabilities, overwrite, num_processes_preprocessing,
                num_processes_segmentation_export, folder_with_segs_from_previous_stage,
                num_parts, part_id
            )
    
    def _predict_multi_view_from_files(self, list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                      output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
                                      save_probabilities: bool = False, overwrite: bool = True,
                                      num_processes_preprocessing: int = 8, num_processes_segmentation_export: int = 8,
                                      folder_with_segs_from_previous_stage: str = None, num_parts: int = 1, part_id: int = 0):
        """多视图预测"""
        
        print("Running multi-view prediction...")
        
        # 解析输入文件
        if isinstance(list_of_lists_or_source_folder, str):
            # 从文件夹读取
            input_folder = list_of_lists_or_source_folder
            case_files = self._parse_multi_view_files(input_folder)
        else:
            # 直接提供文件列表
            case_files = self._group_multi_view_files(list_of_lists_or_source_folder)
        
        # 创建输出文件夹
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
            maybe_mkdir_p(output_folder)
        
        # 处理每个case
        for case_id, files in case_files.items():
            print(f"Processing case: {case_id}")
            
            if 'long' in files and 'short' in files:
                # 多视图case
                self._predict_multi_view_case(case_id, files, output_folder, save_probabilities)
            else:
                # 单视图case
                self._predict_single_view_case(case_id, files, output_folder, save_probabilities)
    
    def _parse_multi_view_files(self, input_folder: str) -> dict:
        """解析多视图文件"""
        case_files = {}
        
        # 扫描文件夹中的所有NIfTI文件
        for file in os.listdir(input_folder):
            if file.endswith('.nii.gz') or file.endswith('.nii'):
                filepath = join(input_folder, file)
                
                # 解析文件名以确定case和视图类型
                if '_long' in file:
                    case_id = file.replace('_long.nii.gz', '').replace('_long.nii', '')
                    if case_id not in case_files:
                        case_files[case_id] = {}
                    case_files[case_id]['long'] = [filepath]
                elif '_short' in file:
                    case_id = file.replace('_short.nii.gz', '').replace('_short.nii', '')
                    if case_id not in case_files:
                        case_files[case_id] = {}
                    case_files[case_id]['short'] = [filepath]
                else:
                    # 单视图文件
                    case_id = file.replace('.nii.gz', '').replace('.nii', '')
                    if case_id not in case_files:
                        case_files[case_id] = {}
                    case_files[case_id]['single'] = [filepath]
        
        return case_files
    
    def _group_multi_view_files(self, file_lists: List[List[str]]) -> dict:
        """将文件列表分组为多视图case"""
        case_files = {}
        
        for i, files in enumerate(file_lists):
            case_id = f"case_{i:04d}"
            
            # 检查文件名以确定视图类型
            long_files = [f for f in files if '_long' in f]
            short_files = [f for f in files if '_short' in f]
            
            if long_files and short_files:
                case_files[case_id] = {
                    'long': long_files,
                    'short': short_files
                }
            else:
                case_files[case_id] = {'single': files}
        
        return case_files
    
    def _predict_multi_view_case(self, case_id: str, files: dict, output_folder: str, save_probabilities: bool):
        """预测多视图case"""
        
        # 加载长轴和短轴数据
        long_data, long_properties = self.preprocess_patient(files['long'])
        short_data, short_properties = self.preprocess_patient(files['short'])
        
        # 构造多视图输入
        multi_view_input = {
            'data_long': torch.from_numpy(long_data[None]).float().to(self.device),
            'data_short': torch.from_numpy(short_data[None]).float().to(self.device)
        }
        
        # 预测
        with torch.no_grad():
            prediction = self.predict_logits_from_preprocessed_data(multi_view_input)
        
        # 后处理
        if isinstance(prediction, tuple):
            prediction = prediction[0]  # 取第一个输出（忽略一致性权重）
        
        if isinstance(prediction, list):
            prediction = prediction[0]  # 取最高分辨率预测
        
        # 转换为numpy
        prediction = prediction.cpu().numpy()
        
        # 应用后处理
        segmentation = self.convert_predicted_logits_to_segmentation_with_correct_shape(
            prediction, long_properties
        )
        
        # 保存结果
        output_file = join(output_folder, f"{case_id}.nii.gz")
        self.save_segmentation_nifti_from_softmax(
            segmentation, output_file, long_properties, 3, None, None, None, force_separate_z=None
        )
        
        if save_probabilities:
            prob_output_file = join(output_folder, f"{case_id}_probabilities.npz")
            np.savez_compressed(prob_output_file, probabilities=prediction)
        
        print(f"Saved: {output_file}")
    
    def _predict_single_view_case(self, case_id: str, files: dict, output_folder: str, save_probabilities: bool):
        """预测单视图case"""
        
        # 使用标准预测流程
        input_files = files.get('single', files.get('long', files.get('short', [])))
        
        # 预处理
        data, properties = self.preprocess_patient(input_files)
        
        # 预测
        prediction = self.predict_logits_from_preprocessed_data(data)
        
        # 后处理
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        
        if isinstance(prediction, list):
            prediction = prediction[0]
        
        # 转换为分割结果
        segmentation = self.convert_predicted_logits_to_segmentation_with_correct_shape(
            prediction, properties
        )
        
        # 保存结果
        output_file = join(output_folder, f"{case_id}.nii.gz")
        self.save_segmentation_nifti_from_softmax(
            segmentation, output_file, properties, 3, None, None, None, force_separate_z=None
        )
        
        if save_probabilities:
            prob_output_file = join(output_folder, f"{case_id}_probabilities.npz")
            np.savez_compressed(prob_output_file, probabilities=prediction)
        
        print(f"Saved: {output_file}")
    
    def predict_logits_from_preprocessed_data(self, data: Union[np.ndarray, torch.Tensor, dict]) -> Union[np.ndarray, torch.Tensor]:
        """从预处理数据预测logits"""
        
        if isinstance(data, dict):
            # 多视图数据
            with torch.no_grad():
                prediction = self.network(data)
            return prediction
        else:
            # 单视图数据
            return super().predict_logits_from_preprocessed_data(data)


def setup_environment():
    """设置环境变量"""
    if 'nnUNet_raw' not in os.environ:
        os.environ['nnUNet_raw'] = str(Path(__file__).parent / 'nnUNet_raw')
    if 'nnUNet_preprocessed' not in os.environ:
        os.environ['nnUNet_preprocessed'] = str(Path(__file__).parent / 'nnUNet_preprocessed')
    if 'nnUNet_results' not in os.environ:
        os.environ['nnUNet_results'] = str(Path(__file__).parent / 'nnUNet_results')


def predict_dataset(model_folder: str, input_folder: str, output_folder: str,
                   folds: Union[str, List[int]] = 'all', save_probabilities: bool = False,
                   num_processes_preprocessing: int = 8, num_processes_segmentation_export: int = 8,
                   device: str = 'cuda', enable_multi_view: bool = True,
                   use_gaussian: bool = True, use_mirroring: bool = True,
                   tile_step_size: float = 0.5):
    """预测整个数据集"""
    
    # 设置设备
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    device_obj = torch.device(device)
    print(f"Using device: {device_obj}")
    
    # 创建预测器
    predictor = MultiScaleFusionPredictor(
        tile_step_size=tile_step_size,
        use_gaussian=use_gaussian,
        use_mirroring=use_mirroring,
        perform_everything_on_gpu=(device == 'cuda'),
        device=device_obj,
        verbose=True,
        enable_multi_view=enable_multi_view
    )
    
    # 初始化模型
    if folds == 'all':
        use_folds = (0, 1, 2, 3, 4)
    elif isinstance(folds, str):
        use_folds = tuple(map(int, folds.split(',')))
    else:
        use_folds = tuple(folds)
    
    predictor.initialize_from_trained_model_folder(
        model_folder, use_folds=use_folds
    )
    
    # 创建输出文件夹
    maybe_mkdir_p(output_folder)
    
    # 执行预测
    predictor.predict_from_files(
        input_folder, output_folder,
        save_probabilities=save_probabilities,
        overwrite=True,
        num_processes_preprocessing=num_processes_preprocessing,
        num_processes_segmentation_export=num_processes_segmentation_export
    )
    
    print(f"Prediction completed! Results saved to: {output_folder}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Predict with Multi-Scale Fusion nnUNet')
    
    parser.add_argument('-m', '--model', type=str, required=True,
                       help='Path to trained model folder')
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='Input folder containing images to predict')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Output folder for predictions')
    parser.add_argument('-f', '--folds', type=str, default='all',
                       help='Folds to use for prediction (e.g., "0,1,2,3,4" or "all")')
    parser.add_argument('-chk', '--checkpoint', type=str, default='checkpoint_final.pth',
                       help='Checkpoint name to use')
    parser.add_argument('--save_probabilities', action='store_true',
                       help='Save prediction probabilities')
    parser.add_argument('--disable_multi_view', action='store_true',
                       help='Disable multi-view prediction')
    parser.add_argument('--disable_tta', action='store_true',
                       help='Disable test time augmentation (mirroring)')
    parser.add_argument('--disable_gaussian', action='store_true',
                       help='Disable Gaussian smoothing for patch aggregation')
    parser.add_argument('-device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to use for prediction')
    parser.add_argument('--tile_step_size', type=float, default=0.5,
                       help='Tile step size for sliding window prediction')
    parser.add_argument('--num_processes_preprocessing', type=int, default=8,
                       help='Number of processes for preprocessing')
    parser.add_argument('--num_processes_segmentation_export', type=int, default=8,
                       help='Number of processes for segmentation export')
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    print(f"Prediction configuration:")
    print(f"  Model: {args.model}")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Folds: {args.folds}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {args.device}")
    print(f"  Multi-view: {not args.disable_multi_view}")
    print(f"  Test-time augmentation: {not args.disable_tta}")
    print(f"  Gaussian smoothing: {not args.disable_gaussian}")
    print(f"  Save probabilities: {args.save_probabilities}")
    
    try:
        predict_dataset(
            model_folder=args.model,
            input_folder=args.input,
            output_folder=args.output,
            folds=args.folds,
            save_probabilities=args.save_probabilities,
            num_processes_preprocessing=args.num_processes_preprocessing,
            num_processes_segmentation_export=args.num_processes_segmentation_export,
            device=args.device,
            enable_multi_view=not args.disable_multi_view,
            use_gaussian=not args.disable_gaussian,
            use_mirroring=not args.disable_tta,
            tile_step_size=args.tile_step_size
        )
        
        print("Prediction completed successfully!")
    
    except Exception as e:
        print(f"Prediction failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()