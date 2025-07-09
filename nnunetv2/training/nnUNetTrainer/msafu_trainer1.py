#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双分支MSAFUNet训练器

该训练器扩展了nnUNetTrainer，支持双分支多尺度注意力融合U-Net（DualBranchMSAFUNet），
用于处理颈动脉斑块超声图像的长轴和短轴。长轴和短轴图像来自不同患者，无法配对，
通过一致性损失鼓励两个分支学习相似的语义特征，提升分割精度和鲁棒性。

主要功能：
1. 使用两个独立的数据加载器分别加载长轴和短轴图像。
2. 双分支网络分别处理长轴和短轴输入，共享结构但独立训练。
3. 使用Dice + 交叉熵损失进行分割监督，结合一致性损失鼓励语义一致性。
4. 支持深度监督，处理多尺度输出。

作者：多尺度融合团队
日期：2025年7月
"""

import torch
import torch.nn as nn
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D as DataLoader
from typing import Dict, Optional, Tuple
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from dynamic_network_architectures.architectures.msafu_net import DualBranchMSAFUNet
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.loss.consistency_loss import SemanticConsistencyLoss
from torch.nn import Conv2d, InstanceNorm2d, LeakyReLU
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.training.dataloading.utils import unpack_dataset
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform

import os
import numpy as np

class MSAFUNetTrainer(nnUNetTrainer):
    """
    双分支MSAFUNet的训练器。

    支持：
    - 分别加载长轴和短轴图像的数据加载器。
    - 双分支网络（DualBranchMSAFUNet），处理长轴和短轴输入。
    - 结合Dice、交叉熵和语义一致性损失进行训练。
    - 支持深度监督，处理多尺度输出。
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.consistency_loss = SemanticConsistencyLoss(loss_type='mse')
        self.consistency_weight = 0.1
        self.batch_size = plans.get('batch_size', 4)
        self.dataloader_train_long: Optional[DataLoader] = None
        self.dataloader_train_short: Optional[DataLoader] = None
        self.enable_deep_supervision = plans.get('enable_deep_supervision', True)
        self.pad_sides = None
        
    def _build_loss(self):
        """构建分割损失（Dice + 交叉熵），适配深度监督开关"""
        dice_loss = MemoryEfficientSoftDiceLoss(apply_nonlin=torch.sigmoid, smooth=1e-5)
        ce_loss = RobustCrossEntropyLoss(weight=torch.tensor([1.0, 5.0, 5.0, 5.0, 10.0], device=self.device), 
                                        ignore_index=self.label_manager.ignore_label)
        
        def combined_loss(output, target):
            if self.enable_deep_supervision and isinstance(output, (list, tuple)):
                weights = [1 / (2 ** i) for i in range(len(output))]
                weights = [w / sum(weights) for w in weights]
                loss = 0.0
                for o, t, w in zip(output, target, weights):
                    loss += w * (dice_loss(o, t) + ce_loss(o, t.long()))
                return loss
            else:
                # 非深度监督，output 和 target 应为 Tensor
                if isinstance(output, (list, tuple)):
                    output = output[0]  # 取第一个尺度
                if isinstance(target, (list, tuple)):
                    target = target[0]  # 取第一个尺度
                return dice_loss(output, target) + ce_loss(output, target.long())

        return combined_loss

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                  arch_init_kwargs: dict,
                                  arch_init_kwargs_req_import: tuple,
                                  num_input_channels: int,
                                  num_output_channels: int,
                                  enable_deep_supervision: bool = True) -> nn.Module:
        """构建双分支MSAFUNet网络。"""
        network = DualBranchMSAFUNet(
            input_channels=num_input_channels,
            n_stages=arch_init_kwargs['n_stages'],
            features_per_stage=arch_init_kwargs['features_per_stage'],
            conv_op=Conv2d,
            kernel_sizes=arch_init_kwargs['kernel_sizes'],
            strides=arch_init_kwargs['strides'],
            n_conv_per_stage=arch_init_kwargs['n_conv_per_stage'],
            num_classes=num_output_channels,
            n_conv_per_stage_decoder=arch_init_kwargs['n_conv_per_stage_decoder'],
            conv_bias=True,
            norm_op=InstanceNorm2d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=LeakyReLU,
            nonlin_kwargs={'inplace': True},
            deep_supervision=enable_deep_supervision
        )
        return network.to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    
    def build_network(self) -> nn.Module:
        """构建网络，添加类型注解以明确返回类型。"""
        if self.configuration_manager.network_arch_class_name is None:
            raise AssertionError("Cannot create network if configuration_manager.network_arch_class_name is None")
        num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager, self.dataset_json)
        return self.build_network_architecture(
            architecture_class_name=self.configuration_manager.network_arch_class_name,
            arch_init_kwargs=self.configuration_manager.network_arch_init_kwargs,
            arch_init_kwargs_req_import=self.configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels=num_input_channels,
            num_output_channels=self.label_manager.num_segmentation_heads,
            enable_deep_supervision=self.enable_deep_supervision
        ).to(self.device)
    def get_training_transforms(self, patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
                                do_dummy_2d_data_aug, use_mask_for_norm=None, is_cascaded=False,
                                foreground_labels=None, regions=None, ignore_label=-1):
        transforms = super().get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm, is_cascaded, foreground_labels, regions, ignore_label
        )
        if not any(isinstance(t, RemoveLabelTansform) for t in transforms.transforms):
            transforms.transforms.append(RemoveLabelTansform(-1, 0))
        return transforms

    def get_validation_transforms(self, deep_supervision_scales, is_cascaded=False, foreground_labels=None,
                                  regions=None, ignore_label=-1):
        transforms = super().get_validation_transforms(
            deep_supervision_scales, is_cascaded, foreground_labels, regions, ignore_label
        )
        if not any(isinstance(t, RemoveLabelTansform) for t in transforms.transforms):
            transforms.transforms.insert(0, RemoveLabelTansform(-1, 0))
        return transforms    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """配置训练和验证数据加载器，支持长轴和短轴独立加载。"""
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        all_train_keys = list(dataset_tr.keys())
        long_axis_keys = [k for k in all_train_keys if 'long' in k.lower()]
        short_axis_keys = [k for k in all_train_keys if 'short' in k.lower()]

        if not long_axis_keys or not short_axis_keys:
            raise ValueError(f"训练数据必须同时包含‘long’和‘short’轴的图像。Long: {len(long_axis_keys)}, Short: {len(short_axis_keys)}")

        dataset_train_long = nnUNetDataset(
            self.preprocessed_dataset_folder, long_axis_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0
        )
        dataset_train_short = nnUNetDataset(
            self.preprocessed_dataset_folder, short_axis_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0
        )
        # 检查原始标签唯一值和 nnUNetDataset 输出
        self.print_to_log_file('检查 dataset_train_long 标签唯一值（原始 .npy 和 nnUNetDataset 输出）:')
        for k in long_axis_keys:
            try:
                # 直接读取原始 .npy 文件
                case_npz_path = os.path.join(self.preprocessed_dataset_folder, k + '.npz')
                case_npy_path = os.path.join(self.preprocessed_dataset_folder, k + '.npy')
                seg_raw = None
                if os.path.isfile(case_npz_path):
                    seg_raw = np.load(case_npz_path)['seg']
                elif os.path.isfile(case_npy_path):
                    seg_raw = np.load(case_npy_path)
                if seg_raw is not None:
                    unique_raw = np.unique(seg_raw)
                    self.print_to_log_file(f"long {k} 原始: {unique_raw.tolist()}")
                    if -1 in unique_raw:
                        self.print_to_log_file(f"检测到 -1 标签于 long {k} 原始，停止后续检查。")
                        break
                # nnUNetDataset 输出
                _, seg, _ = dataset_train_long.load_case(k)
                unique_vals = np.unique(seg)
                self.print_to_log_file(f"long {k} nnUNetDataset: {unique_vals.tolist()}")
                if -1 in unique_vals:
                    self.print_to_log_file(f"检测到 -1 标签于 long {k} nnUNetDataset，停止后续检查。")
                    break
            except Exception as e:
                self.print_to_log_file(f"long {k} 加载失败: {e}")
        self.print_to_log_file('检查 dataset_train_short 标签唯一值（原始 .npy 和 nnUNetDataset 输出）:')
        for k in short_axis_keys:
            try:
                case_npz_path = os.path.join(self.preprocessed_dataset_folder, k + '.npz')
                case_npy_path = os.path.join(self.preprocessed_dataset_folder, k + '.npy')
                seg_raw = None
                if os.path.isfile(case_npz_path):
                    seg_raw = np.load(case_npz_path)['seg']
                elif os.path.isfile(case_npy_path):
                    seg_raw = np.load(case_npy_path)
                if seg_raw is not None:
                    unique_raw = np.unique(seg_raw)
                    self.print_to_log_file(f"short {k} 原始: {unique_raw.tolist()}")
                    if -1 in unique_raw:
                        self.print_to_log_file(f"检测到 -1 标签于 short {k} 原始，停止后续检查。")
                        break
                _, seg, _ = dataset_train_short.load_case(k)
                unique_vals = np.unique(seg)
                self.print_to_log_file(f"short {k} nnUNetDataset: {unique_vals.tolist()}")
                if -1 in unique_vals:
                    self.print_to_log_file(f"检测到 -1 标签于 short {k} nnUNetDataset，停止后续检查。")
                    break
            except Exception as e:
                self.print_to_log_file(f"short {k} 加载失败: {e}")
        total_batch_size = self.batch_size
        batch_size_long = total_batch_size // 2 + (total_batch_size % 2)
        batch_size_short = total_batch_size // 2

        patch_size = self.configuration_manager.patch_size

        # 添加数据增强
        deep_supervision_scales = self._get_deep_supervision_scales()
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label
        )
        
        self.print_to_log_file(f"Training transforms: {tr_transforms.transforms}")
        self.print_to_log_file(f"Validation transforms: {val_transforms.transforms}")

        # 使用多线程增强
        allowed_num_processes = get_allowed_n_proc_DA()
        seeds = [None] * allowed_num_processes

        self.print_to_log_file(f"Using {allowed_num_processes} processes for data loading, seeds: {seeds}")
        
        self.dataloader_train_long = MultiThreadedAugmenter(
            DataLoader(
                data=dataset_train_long, 
                batch_size=batch_size_long, 
                patch_size=patch_size, 
                final_patch_size=patch_size,
                label_manager=self.label_manager, 
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, 
                pad_sides=self.pad_sides,
                probabilistic_oversampling=False, 
                transforms=tr_transforms  
            ),
            transform=None,
            num_processes=allowed_num_processes,
            num_cached_per_queue=max(2, allowed_num_processes // 2),
            seeds=seeds,
            pin_memory=False
        )
        self.dataloader_train_short = MultiThreadedAugmenter(
            DataLoader(
                data=dataset_train_short, 
                batch_size=batch_size_short, 
                patch_size=patch_size, 
                final_patch_size=patch_size,
                label_manager=self.label_manager, 
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, 
                pad_sides=self.pad_sides,
                probabilistic_oversampling=False, 
                transforms=tr_transforms
            ),
            transform=None,
            num_processes=allowed_num_processes,
            num_cached_per_queue=max(2, allowed_num_processes // 2),
            seeds=seeds,
            pin_memory=False
        )
        dl_val = MultiThreadedAugmenter(
            DataLoader(
                data=dataset_val, 
                batch_size=total_batch_size, 
                patch_size=patch_size, 
                final_patch_size=patch_size,
                label_manager=self.label_manager, 
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, 
                pad_sides=self.pad_sides,
                probabilistic_oversampling=False,
                transforms=val_transforms   
            ),
            transform=None, 
            num_processes=max(1, allowed_num_processes // 2),
            num_cached_per_queue=max(2, allowed_num_processes // 4),
            seeds=[None] * max(1, allowed_num_processes // 2),
            pin_memory=False  # 禁用 pin_memory
        )
        self.print_to_log_file(f"dataloader_train_long: {self.dataloader_train_long}")
        self.print_to_log_file(f"dataloader_train_short: {self.dataloader_train_short}")
        self.print_to_log_file(f"dl_val: {dl_val}")

        return self.dataloader_train_long, self.dataloader_train_short, dl_val
        
    def on_train_start(self):
        """重写 on_train_start，处理三个数据加载器并初始化训练。"""
        if self.unpack_dataset and self.local_rank == 0:
            self.print_to_log_file('unpacking dataset...')
            unpack_dataset(self.preprocessed_dataset_folder, unpack_segmentation=True,
                          overwrite_existing=False, num_processes=2, verify_npy=True)
            self.print_to_log_file('unpacking done...')
        
        # 获取三个数据加载器
        self.dataloader_train_long, self.dataloader_train_short, self.dataloader_val = self.get_dataloaders()
        self.dataloader_train = self.dataloader_train_long  # 兼容父类
        self.dataloader_val = self.dataloader_val
        
        # 初始化网络、优化器、学习率调度器等
        self.network = self.build_network()
        self.print_to_log_file(f"Network type: {type(self.network)}")
        if self._do_i_compile():
            self.print_to_log_file('Using torch.compile...')
            self.network = torch.compile(self.network)
        if self.is_ddp:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            self.network = torch.nn.parallel.DistributedDataParallel(self.network, device_ids=[self.local_rank])
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        self.initialize()

        self.dataloader_train_long, self.dataloader_train_short, self.dataloader_val = self.get_dataloaders()
        self.print_to_log_file(f"dataloader_train_long: {self.dataloader_train_long}")
        self.print_to_log_file(f"dataloader_train_short: {self.dataloader_train_short}")
        self.print_to_log_file(f"dataloader_val: {self.dataloader_val}")

    def train_step(self, batch: Dict[str, dict]) -> dict:
        """执行训练步骤，复用 nnUNetTrainer.train_step，计算分割损失和一致性损失。"""
        self.optimizer.zero_grad(set_to_none=True)

        batch_long = batch['long']
        batch_short = batch['short']
        
        # 使用 nnUNetTrainer.train_step 处理长轴和短轴
        with torch.amp.autocast(device_type='cuda', enabled=True):
            # 前向传播
            output_long, output_short, features_long, features_short = self.network(batch_long['data'].to(self.device, non_blocking=True),
                                                                                   batch_short['data'].to(self.device, non_blocking=True))
            
            # 调用 nnUNetTrainer.train_step 计算分割损失
            loss_seg_long = super().train_step({'data': batch_long['data'], 'target': batch_long['target']})['loss']
            loss_seg_short = super().train_step({'data': batch_short['data'], 'target': batch_short['target']})['loss']
            
            # 计算一致性损失
            loss_consistency = self.consistency_loss(features_long, features_short)
            
            # 合并总损失
            total_loss = loss_seg_long + loss_seg_short + self.consistency_weight * loss_consistency

        self.grad_scaler.scale(total_loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        return {'loss': total_loss.detach().cpu().numpy()}
        
    def validation_step(self, batch: dict) -> dict:
        """执行验证步骤，复用 nnUNetTrainer.validation_step，计算长轴和短轴的平均损失。"""
        # 调试：记录 batch 数据格式
        
        with torch.amp.autocast(device_type='cuda', enabled=True):
            # 前向传播
            output_long, output_short, _, _ = self.network(batch['data'].to(self.device, non_blocking=True),
                                                          batch['data'].to(self.device, non_blocking=True))
            
            # 使用 nnUNetTrainer.validation_step 计算损失
            loss_long = super().validation_step({'data': batch['data'], 'target': batch['target']})['loss']
            loss_short = super().validation_step({'data': batch['data'], 'target': batch['target']})['loss']
            loss = (loss_long + loss_short) / 2

        return {
            'loss': loss.detach().cpu().numpy(),
            'loss_long': loss_long.detach().cpu().numpy(),
            'loss_short': loss_short.detach().cpu().numpy()
        }
        
    def run_training(self) -> None:
        """运行训练过程，协调训练和验证循环。"""
        self.on_train_start()
        self.print_to_log_file("Testing data loading...")
        try:
            batch_long = next(iter(self.dataloader_train_long))
            batch_short = next(iter(self.dataloader_train_short))
            batch_val = next(iter(self.dataloader_val))
            self.print_to_log_file(f"Sample batch_long: {batch_long['data'].shape}")
            self.print_to_log_file(f"Sample batch_short: {batch_short['data'].shape}")
            self.print_to_log_file(f"Sample batch_val: {batch_val['data'].shape}")
        except Exception as e:
            self.print_to_log_file(f"Data loading error: {str(e)}")
            raise e

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_losses = []

            if self.dataloader_train_long and self.dataloader_train_short:
                long_iter = iter(self.dataloader_train_long)
                short_iter = iter(self.dataloader_train_short)
                total_iters = self.num_iterations_per_epoch

                for _ in range(total_iters):
                    try:
                        batch_long = next(long_iter)
                    except StopIteration:
                        long_iter = iter(self.dataloader_train_long)
                        batch_long = next(long_iter)
                    
                    try:
                        batch_short = next(short_iter)
                    except StopIteration:
                        short_iter = iter(self.dataloader_train_short)
                        batch_short = next(short_iter)
                    
                    loss = self.train_step({'long': batch_long, 'short': batch_short})
                    train_losses.append(loss['loss'])
            
            self.on_train_epoch_end(train_losses)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_losses = []
                val_losses_long = []
                val_losses_short = []
                for batch in self.dataloader_val:
                    loss = self.validation_step(batch)
                    val_losses.append(loss['loss'])
                    val_losses_long.append(loss['loss_long'])
                    val_losses_short.append(loss['loss_short'])
                self.on_validation_epoch_end(val_losses)
                self.print_to_log_file(f"Validation losses - Total: {np.mean(val_losses):.4f}, "
                                     f"Long: {np.mean(val_losses_long):.4f}, Short: {np.mean(val_losses_short):.4f}")

            self.on_epoch_end()
        self.on_train_end()
