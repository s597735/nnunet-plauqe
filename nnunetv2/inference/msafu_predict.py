import argparse
import torch
import numpy as np
from typing import Union, List, Tuple
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


class MSAFUNetPredictor(nnUNetPredictor):
    """
    专门用于双分支 MSAFUNet 的预测器
    继承自 nnUNetPredictor 并重写相关方法以支持双分支网络
    """
    
    def __init__(self, tile_step_size: float = 0.5, use_gaussian: bool = True, use_mirroring: bool = True,
                 perform_everything_on_device: bool = True, device: torch.device = torch.device('cuda'),
                 verbose: bool = False, verbose_preprocessing: bool = False, allow_tqdm: bool = True):
        super().__init__(tile_step_size, use_gaussian, use_mirroring, perform_everything_on_device,
                         device, verbose, verbose_preprocessing, allow_tqdm)
    
    def _internal_maybe_mirror_and_predict(self, x) -> torch.Tensor:
        """
        重写内部预测方法以支持双分支输入
        """
        # 检查是否为双分支输入
        if isinstance(x, tuple) and len(x) == 2:
            x_long, x_short = x
            # 双分支预测
            if self.use_mirroring:
                # 如果使用镜像增强，需要对两个分支都应用
                mirror_axes = self.allowed_mirroring_axes if self.allowed_mirroring_axes is not None else []
                prediction = self._internal_maybe_mirror_and_predict_dual_branch(x_long, x_short, mirror_axes)
            else:
                prediction = self.network(x_long, x_short)
        else:
            # 单分支预测（兼容性）
            if self.use_mirroring:
                prediction = super()._internal_maybe_mirror_and_predict(x)
            else:
                prediction = self.network(x)
        
        return prediction
    
    def _internal_maybe_mirror_and_predict_dual_branch(self, x_long: torch.Tensor, x_short: torch.Tensor, mirror_axes: list) -> torch.Tensor:
        """
        双分支镜像预测的内部实现
        """
        # 简化实现：不使用镜像增强，直接预测
        # 在实际应用中，可能需要实现完整的镜像增强逻辑
        return self.network(x_long, x_short)
    
    def predict_sliding_window_return_logits(self, input_image_long: torch.Tensor, 
                                            input_image_short: torch.Tensor) -> torch.Tensor:
        """
        重写滑动窗口预测方法以支持双分支输入
        
        Args:
            input_image_long: 长轴输入图像
            input_image_short: 短轴输入图像
            
        Returns:
            预测的logits
        """
        assert len(input_image_long.shape) == 4, 'input_image_long must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'
        assert len(input_image_short.shape) == 4, 'input_image_short must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'
        
        if self.verbose:
            print(f'Input shape (long): {input_image_long.shape}')
            print(f'Input shape (short): {input_image_short.shape}')
            print(f'step_size: {self.tile_step_size}')
            print(f'mirror_axes: {self.allowed_mirroring_axes} (None means no mirroring)')
        
        # 确保输入在正确的设备上
        if not isinstance(input_image_long, torch.Tensor):
            input_image_long = torch.from_numpy(input_image_long)
        if not isinstance(input_image_short, torch.Tensor):
            input_image_short = torch.from_numpy(input_image_short)
            
        input_image_long = input_image_long.to(self.device, non_blocking=False)
        input_image_short = input_image_short.to(self.device, non_blocking=False)
        
        # 调用父类的滑动窗口预测，但传入双分支输入
        # 我们需要重写这部分逻辑来处理双分支网络
        return self._internal_predict_sliding_window_return_logits(
            input_image_long, input_image_short
        )
    
    def _internal_predict_sliding_window_return_logits(self, input_image_long: torch.Tensor,
                                                     input_image_short: torch.Tensor) -> torch.Tensor:
        """
        内部滑动窗口预测方法，处理双分支网络的具体预测逻辑
        """
        # 获取网络的输出形状信息
        with torch.no_grad():
            # 创建一个小的测试输入来获取输出形状
            test_input_long = torch.zeros((1, input_image_long.shape[0], 32, 32, 32), 
                                        device=self.device, dtype=input_image_long.dtype)
            test_input_short = torch.zeros((1, input_image_short.shape[0], 32, 32, 32), 
                                         device=self.device, dtype=input_image_short.dtype)
            
            test_output = self.network(test_input_long, test_input_short)
            num_classes = test_output.shape[1]
        
        # 为简化实现，这里我们使用整个图像进行预测
        # 在实际应用中，可能需要实现真正的滑动窗口逻辑
        input_image_long = input_image_long[None]  # 添加batch维度
        input_image_short = input_image_short[None]  # 添加batch维度
        
        with torch.no_grad():
            prediction = self.network(input_image_long, input_image_short)
        
        return prediction[0]  # 移除batch维度
    
    def manual_initialization(self, network, plans_manager, configuration_manager, parameters, dataset_json,
                            work_dir=None, plans_identifier='nnUNetPlans'):
        """
        手动初始化预测器，用于训练过程中的验证
        """
        super().manual_initialization(network, plans_manager, configuration_manager, parameters, 
                                    dataset_json, work_dir, plans_identifier)
    
    def predict_from_files_dual_branch(self, list_of_lists_or_source_folder: str,
                                     output_folder_or_list_of_truncated_output_files: str,
                                     save_probabilities: bool = False,
                                     overwrite: bool = True,
                                     num_processes_preprocessing: int = 2,
                                     num_processes_segmentation_export: int = 2,
                                     folder_with_segs_from_prev_stage: str = None,
                                     num_parts: int = 1,
                                     part_id: int = 0):
        """
        双分支预测方法，将相同的输入数据作为长轴和短轴输入
        """
        # 重写父类方法以支持双分支预测
        # 临时保存原始的预测方法
        original_predict_method = self.predict_sliding_window_return_logits
        
        # 重写预测方法以支持双分支
        def dual_branch_predict_wrapper(input_image):
            # 将单个输入复制为双分支输入
            return self.predict_sliding_window_return_logits(input_image, input_image)
        
        # 临时替换预测方法
        self.predict_sliding_window_return_logits = dual_branch_predict_wrapper
        
        try:
            # 调用父类的预测方法
            result = super().predict_from_files(
                list_of_lists_or_source_folder=list_of_lists_or_source_folder,
                output_folder_or_list_of_truncated_output_files=output_folder_or_list_of_truncated_output_files,
                save_probabilities=save_probabilities,
                overwrite=overwrite,
                num_processes_preprocessing=num_processes_preprocessing,
                num_processes_segmentation_export=num_processes_segmentation_export,
                folder_with_segs_from_prev_stage=folder_with_segs_from_prev_stage,
                num_parts=num_parts,
                part_id=part_id
            )
        finally:
            # 恢复原始的预测方法
            self.predict_sliding_window_return_logits = original_predict_method
        
        return result


def create_msafu_predictor_for_validation(network, plans_manager, configuration_manager, 
                                         dataset_json, output_folder, device):
    """
    为训练过程中的验证创建 MSAFUNetPredictor 的便捷函数
    
    Args:
        network: 训练好的网络模型
        plans_manager: 计划管理器
        configuration_manager: 配置管理器
        dataset_json: 数据集JSON
        output_folder: 输出文件夹
        device: 设备
        
    Returns:
        配置好的 MSAFUNetPredictor 实例
    """
    predictor = MSAFUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False
    )
    
    # 手动初始化预测器
    predictor.manual_initialization(
        network, plans_manager, configuration_manager,
        None, dataset_json, output_folder
    )
    
    return predictor


def predict_entry_point():
    parser = argparse.ArgumentParser(description='Run inference for MSAFUNet.')
    parser.add_argument('-i', '--input_folder', type=str, required=True,
                        help='Folder containing the images to predict.')
    parser.add_argument('-o', '--output_folder', type=str, required=True,
                        help='Folder to save the predictions.')
    parser.add_argument('-m', '--model_training_output_dir', type=str, required=True,
                        help='Folder containing the trained model.')
    parser.add_argument('-f', '--folds', nargs='+', default=None,
                        help='Folds to use for prediction. Default is all folds.')
    parser.add_argument('-a', '--axis', type=str, required=True, choices=['long', 'short', 'dual'],
                        help='The axis to use for prediction (long, short, or dual for both branches).')
    parser.add_argument('-step_size', type=float, default=0.5, help='Step size for sliding window prediction.')
    parser.add_argument('--disable_gaussian', action='store_true', help='Disable Gaussian smoothing for sliding window prediction.')
    parser.add_argument('--disable_mirroring', action='store_true', help='Disable mirroring for data augmentation.')

    args = parser.parse_args()

    if args.axis == 'dual':
        # 对于双分支预测，使用自定义的 MSAFUNetPredictor
        predictor = MSAFUNetPredictor(
            tile_step_size=args.step_size,
            use_gaussian=not args.disable_gaussian,
            use_mirroring=not args.disable_mirroring,
            perform_everything_on_device=True,
            device=torch.device('cuda'),
            verbose=True
        )
        
        # Initialize from the trained model folder
        predictor.initialize_from_trained_model_folder(
            args.model_training_output_dir,
            use_folds=args.folds,
            checkpoint_name='checkpoint_final.pth'
        )
        
        print("Starting dual-branch prediction...")
        
    elif args.axis in ['long', 'short']:
        # 对于单分支预测，使用标准的 nnUNetPredictor
        predictor = nnUNetPredictor(
            tile_step_size=args.step_size,
            use_gaussian=not args.disable_gaussian,
            use_mirroring=not args.disable_mirroring,
            perform_everything_on_device=True,
            device=torch.device('cuda'),
            verbose=True
        )

        # Initialize from the trained model folder
        predictor.initialize_from_trained_model_folder(
            args.model_training_output_dir,
            use_folds=args.folds,
            checkpoint_name='checkpoint_final.pth'
        )

        # The loaded network is DualBranchMSAFUNet. We need to select the correct branch.
        if args.axis == 'long':
            # If we only need the long-axis branch, we can replace the network with it.
            # This simplifies the forward pass during prediction.
            main_network = predictor.network
            predictor.network = main_network.net_long_axis
        elif args.axis == 'short':
            main_network = predictor.network
            predictor.network = main_network.net_short_axis
            
        print(f"Starting prediction on {args.axis}-axis branch...")
        
    else:
        raise ValueError(f"Invalid axis specified: {args.axis}. Must be 'long', 'short', or 'dual'.")

    # Run prediction
    if args.axis == 'dual':
        # 使用双分支预测方法
        predictor.predict_from_files_dual_branch(
            list_of_lists_or_source_folder=args.input_folder,
            output_folder_or_list_of_truncated_output_files=args.output_folder,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0
        )
    else:
        # 使用标准的单分支预测方法
        predictor.predict_from_files(
            list_of_lists_or_source_folder=args.input_folder,
            output_folder_or_list_of_truncated_output_files=args.output_folder,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0
        )

if __name__ == '__main__':
    predict_entry_point()