import torch
import numpy as np
from copy import deepcopy
from batchgenerators.utilities.file_and_folder_operations import join

# 导入nnUNetv2的2D训练器作为基类
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D

class CarotidContinualTrainer2D(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), ewc_lambda=5000, distill_temp=2.0):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.ewc_lambda = ewc_lambda  # EWC正则化强度
        self.distill_temp = distill_temp  # 知识蒸馏温度参数
        self.fisher_information = None  # 用于EWC的Fisher信息矩阵
        self.old_params = None  # 存储旧模型参数
        self.teacher_model = None  # 教师模型
        self.stage = None  # 训练阶段标识
        
        # 2D训练特定参数调整
        self.initial_lr = 1e-3  # 2D模型通常使用更小的学习率
        self.weight_decay = 3e-5  # 权重衰减
        self.num_epochs = 500  # 2D模型通常需要更多的训练轮次
        
    def initialize(self):
        """初始化训练器，加载预训练的长轴模型作为起点"""
        if not self.was_initialized:
            super().initialize()
            
            # 如果是持续学习阶段，加载预训练模型
            if self.stage == 'continual':
                self.load_pretrained_longitudinal_model()
                
            self.was_initialized = True
        else:
            self.print_to_log_file("WARNING: 训练器已经初始化过，跳过重复初始化")
            
    def load_pretrained_longitudinal_model(self):
        """加载预训练的长轴模型"""
        # 2D模型路径
        pretrained_model_path = join(self.output_folder_base.replace('CarotidContinualTrainer2D', 'nnUNetTrainer'),
                                     f'fold_{self.fold}', 'checkpoint_final.pth')
        
        self.print_to_log_file(f"加载预训练模型: {pretrained_model_path}")
        
        if not self.was_initialized:
            self.initialize()
            
        try:
            saved_model = torch.load(pretrained_model_path, map_location=self.device)
            self.network.load_state_dict(saved_model['network_weights'])
            
            # 创建教师模型（冻结的长轴模型）
            self.teacher_model = deepcopy(self.network)
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
            
            # 保存旧模型参数用于EWC
            self.old_params = {name: param.clone().detach() 
                              for name, param in self.network.named_parameters()}
            
            self.print_to_log_file("预训练长轴模型加载成功，准备进行持续学习...")
        except Exception as e:
            self.print_to_log_file(f"加载预训练模型失败: {e}")
            raise
        
    def get_basic_generators(self):
        """获取2D数据生成器"""
        self.load_dataset()
        self.do_split()

        # 确保使用2D数据加载器
        if self.threeD:
            self.print_to_log_file("警告：检测到3D配置，但强制使用2D数据加载器")
            
        # 使用2D数据加载器
        dl_tr = nnUNetDataLoader2D(
            self.dataset[self.tr_keys],
            self.batch_size,
            self.patch_size,
            self.final_patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=self.pad_all_sides,
            probabilistic_oversampling=self.probabilistic_oversampling
        )
        
        dl_val = nnUNetDataLoader2D(
            self.dataset[self.val_keys],
            self.batch_size,
            self.patch_size,
            self.final_patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=self.pad_all_sides,
            probabilistic_oversampling=self.probabilistic_oversampling
        )
        
        return dl_tr, dl_val
        
    def compute_fisher_information(self, dataloader, num_samples=200):
        """计算Fisher信息矩阵，用于EWC正则化"""
        self.network.eval()
        fisher_info = {name: torch.zeros_like(param) 
                      for name, param in self.network.named_parameters() if param.requires_grad}
        
        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break
                
            x = batch['data'].to(self.device)
            y = batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            output = self.network(x)
            loss = self.loss(output, y)
            loss.backward()
            
            # 累积梯度的平方作为Fisher信息的估计
            for name, param in self.network.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
                    
            sample_count += x.shape[0]
            
        # 归一化Fisher信息
        for name in fisher_info:
            fisher_info[name] /= sample_count
            
        self.fisher_information = fisher_info
        self.network.train()
        self.print_to_log_file(f"Fisher信息矩阵计算完成，使用了{sample_count}个样本")
        
    def compute_ewc_loss(self):
        """计算EWC损失，防止遗忘长轴任务"""
        if self.fisher_information is None or self.old_params is None:
            return 0
            
        ewc_loss = 0
        for name, param in self.network.named_parameters():
            if name in self.fisher_information and name in self.old_params:
                ewc_loss += torch.sum(self.fisher_information[name] * 
                                     (param - self.old_params[name]) ** 2)
                
        return self.ewc_lambda * ewc_loss
        
    def compute_distillation_loss(self, student_output, teacher_output):
        """计算知识蒸馏损失"""
        # 处理深度监督输出
        if isinstance(student_output, list) and isinstance(teacher_output, list):
            # 只使用最高分辨率的输出进行蒸馏
            student_output = student_output[0]
            teacher_output = teacher_output[0]
            
        # 使用KL散度作为蒸馏损失
        student_logits = student_output / self.distill_temp
        teacher_logits = teacher_output / self.distill_temp
        
        # 对logits应用softmax
        student_probs = torch.nn.functional.softmax(student_logits, dim=1)
        teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=1)
        
        # 计算KL散度
        distill_loss = torch.nn.functional.kl_div(
            torch.log(student_probs + 1e-10), 
            teacher_probs, 
            reduction='batchmean'
        ) * (self.distill_temp ** 2)
        
        return distill_loss
        
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """重写迭代函数，加入持续学习组件"""
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad()
        
        # 前向传播
        output = self.network(data)
        
        # 计算任务损失
        task_loss = self.loss(output, target)
        
        # 如果是短轴数据，添加知识蒸馏
        is_short_axis = data_dict.get('is_short_axis', False)
        if is_short_axis and self.teacher_model is not None:
            with torch.no_grad():
                teacher_output = self.teacher_model(data)
            distill_loss = self.compute_distillation_loss(output, teacher_output)
            task_loss = 0.7 * task_loss + 0.3 * distill_loss
        
        # 添加EWC损失防止遗忘
        ewc_loss = self.compute_ewc_loss()
        
        # 总损失
        loss = task_loss + ewc_loss
        
        # 反向传播
        if do_backprop:
            if self.grad_scaler is not None:
                with autocast(self.device.type, enabled=True):
                    # 重新计算一次前向传播，以便使用混合精度训练
                    output = self.network(data)
                    task_loss = self.loss(output, target)
                    if is_short_axis and self.teacher_model is not None:
                        with torch.no_grad():
                            teacher_output = self.teacher_model(data)
                        distill_loss = self.compute_distillation_loss(output, teacher_output)
                        task_loss = 0.7 * task_loss + 0.3 * distill_loss
                    loss = task_loss + ewc_loss
                
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()
                
        # 在线评估
        if run_online_evaluation:
            self.run_online_evaluation(output, target)
            
        # 更新学习率
        self.lr_scheduler.step(self.current_epoch)
            
        return loss.detach().cpu().numpy()
        
    def on_epoch_end(self):
        """每个epoch结束时的操作"""
        super().on_epoch_end()
        
        # 定期更新Fisher信息矩阵
        if self.current_epoch % 10 == 0 and self.stage == 'continual':
            self.print_to_log_file("更新Fisher信息矩阵...")
            self.compute_fisher_information(self.dataloader_train)
            
    def run_training(self):
        """运行训练流程"""
        if not self.was_initialized:
            self.initialize()

        # 确保数据加载器已初始化
        if self.dataloader_train is None:
            self.dataloader_train, self.dataloader_val = self.get_basic_generators()
            self.print_to_log_file(f"数据加载器初始化完成，训练集大小: {len(self.dataloader_train)}, 验证集大小: {len(self.dataloader_val)}")
            
        # 如果是持续学习阶段，计算初始Fisher信息矩阵
        if self.stage == 'continual' and self.fisher_information is None:
            self.print_to_log_file("计算初始Fisher信息矩阵...")
            self.compute_fisher_information(self.dataloader_train)
            
        # 运行训练循环
        self.print_to_log_file(f"开始训练，总共{self.num_epochs}个epoch")
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.print_to_log_file(f"\nEpoch {epoch}")
            
            # 训练一个epoch
            self.network.train()
            train_losses = []
            for _ in range(self.num_iterations_per_epoch):
                l = self.run_iteration(self.dataloader_train, True)
                train_losses.append(l)
            self.print_to_log_file(f"训练损失: {np.mean(train_losses):.5f}")
            
            # 验证
            self.network.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(self.num_val_iterations_per_epoch):
                    l = self.run_iteration(self.dataloader_val, False, True)
                    val_losses.append(l)
            self.print_to_log_file(f"验证损失: {np.mean(val_losses):.5f}")
            
            # 保存检查点
            if not self.disable_checkpointing and (epoch % self.save_every == 0 or epoch == self.num_epochs - 1):
                self.save_checkpoint(join(self.output_folder, f"checkpoint_{epoch}.pth"))
                
            # epoch结束处理
            self.on_epoch_end()
            
        # 保存最终模型
        if not self.disable_checkpointing:
            self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
            
        self.print_to_log_file("训练完成!")
        
    def save_checkpoint(self, filename):
        """保存检查点"""
        if self.local_rank == 0:
            checkpoint = {
                'network_weights': self.network.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                'logging': self.logger.get_checkpoint(),
                'current_epoch': self.current_epoch,
                'init_args': self.my_init_kwargs,
                'trainer_name': self.__class__.__name__,
                'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                # 保存持续学习相关参数
                'ewc_lambda': self.ewc_lambda,
                'distill_temp': self.distill_temp,
                'stage': self.stage,
                'fisher_information': self.fisher_information,
                'old_params': self.old_params
            }
            torch.save(checkpoint, filename)
            self.print_to_log_file(f"模型已保存到 {filename}")
            
    def load_checkpoint(self, filename):
        """加载检查点"""
        if not self.was_initialized:
            self.initialize()
            
        checkpoint = torch.load(filename, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_weights'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None and checkpoint['grad_scaler_state'] is not None:
            self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
        self.logger.load_checkpoint(checkpoint['logging'])
        self.current_epoch = checkpoint['current_epoch']
        self.inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes']
        
        # 加载持续学习相关参数
        if 'ewc_lambda' in checkpoint:
            self.ewc_lambda = checkpoint['ewc_lambda']
        if 'distill_temp' in checkpoint:
            self.distill_temp = checkpoint['distill_temp']
        if 'stage' in checkpoint:
            self.stage = checkpoint['stage']
        if 'fisher_information' in checkpoint:
            self.fisher_information = checkpoint['fisher_information']
        if 'old_params' in checkpoint:
            self.old_params = checkpoint['old_params']
            
        self.print_to_log_file(f"模型已从 {filename} 加载")