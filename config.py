import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-0.5B"
    trust_remote_code: bool = True
    
    # 数据集配置
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"  # GSM8K数据集的配置
    dataset_split: str = "train"
    max_length: int = 512
    max_samples: Optional[int] = 1000  # 为了快速验证，限制样本数量
    
    # GRPO训练配置
    learning_rate: float = 5e-6
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2  # Apple Silicon优化
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    
    # GRPO特定参数
    beta: float = 0.1  # KL散度惩罚系数
    grpo_epochs: int = 1  # 每个batch的GRPO更新次数
    
    # 优化器配置
    optim: str = "adamw_torch"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # 设备配置 (Apple Silicon优化)
    device_map: str = "auto"
    torch_dtype: str = "float32"  # 改为float32以避免精度问题
    use_mps: bool = torch.backends.mps.is_available()
    
    # 输出配置
    output_dir: str = "./outputs"
    logging_dir: str = "./logs"
    
    # Reward模型配置
    reward_model_name: str = "Qwen/Qwen2.5-0.5B"  # 使用相同模型作为reward模型
    
    # 其他配置
    seed: int = 42
    dataloader_num_workers: int = 0  # Apple Silicon上设为0避免问题
    remove_unused_columns: bool = False
    
    def get_device(self):
        """获取训练设备"""
        if self.use_mps and torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def get_torch_dtype(self):
        """获取torch数据类型"""
        if self.torch_dtype == "float16":
            return torch.float16
        elif self.torch_dtype == "bfloat16":
            return torch.bfloat16
        else:
            return torch.float32 