import torch
import numpy as np
from copy import deepcopy
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class CarotidContinualTrainer(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, ewc_lambda=5000, distill_temp=2.0):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                         unpack_data, deterministic, fp16)
        self.ewc_lambda = ewc_lambda  # EWC正则化强度
        self.distill_temp = distill_temp  # 知识蒸馏温度参数
        self.fisher_information = None  # 用于EWC的Fisher信息矩阵
        self.old_params = None  # 存储旧模型参数
        self.teacher_model = None  # 教师模型
        
    def initialize(self, training=True, force_load_plans=False):
        """初始化训练器，加载预训练的长轴模型作为起点"""
        super().initialize(training, force_load_plans)
        
        # 如果是持续学习阶段，加载预训练模型
        if self.stage == 'continual':
            self.load_pretrained_longitudinal_model()
            
    def load_pretrained_longitudinal_model(self):
        """加载预训练的长轴模型"""
        # 假设预训练模型路径
        pretrained_model_path = 'e:/SWQ/nnUNet/RESULTS/nnUNet/3d_fullres/TaskXXX_CarotidLongitudinal/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_final_checkpoint.model'
        
        if not self.was_initialized:
            self.initialize(training=True)
            
        saved_model = torch.load(pretrained_model_path, map_location=torch.device('cuda'))
        self.network.load_state_dict(saved_model['state_dict'])
        
        # 创建教师模型（冻结的长轴模型）
        self.teacher_model = deepcopy(self.network)
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
        # 保存旧模型参数用于EWC
        self.old_params = {name: param.clone().detach() 
                          for name, param in self.network.named_parameters()}
        
        print("预训练长轴模型加载成功，准备进行持续学习...")
        
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
        print(f"Fisher信息矩阵计算完成，使用了{sample_count}个样本")
        
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
            loss.backward()
            self.optimizer.step()
            
        # 在线评估
        if run_online_evaluation:
            self.run_online_evaluation(output, target)
            
        return loss.detach().cpu().numpy()
        
    def train_step(self, batch_idx):
        """训练步骤"""
        # 原始训练步骤
        super().train_step(batch_idx)
        
        # 定期更新Fisher信息矩阵
        if batch_idx % 1000 == 0 and self.stage == 'continual':
            print("更新Fisher信息矩阵...")
            self.compute_fisher_information(self.tr_gen)