import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    trust_remote_code: bool = True
    
    # 数据集配置
    dataset_name: str = "custom_math" # "openai/gsm8k"
    dataset_config: str = "main"  # GSM8K数据集的配置
    dataset_split: str = "train"
    max_length: int = 512  # 减少长度，适合简单的加减法
    max_new_tokens: int = 32  # 减少生成长度，适合简单答案
    max_samples: Optional[int] = 200  # 增加样本数量用于简单数学训练
    
    # 自定义数学数据集配置
    custom_math_size: int = 500  # 自定义数据集大小
    max_number: int = 10  # 数字范围：0-10
    
    # GRPO训练配置
    # learning_rate: float = 5e-6
    learning_rate: float = 5e-6  # 稍微提高学习率，适合简单任务
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    logging_steps: int = 1
    
    # GRPO特定参数
    beta: float = 0.1  # KL散度惩罚系数
    grpo_epochs: int = 1  # 每个batch的GRPO更新次数
    group_size: int = 8
    clip_epsilon: float = 0.2  # PPO clipping参数
    kl_coeff: float = 0.01  # KL散度正则化系数
    
    # 优化器配置
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # 设备配置 (Apple Silicon优化)
    device_map: str = "auto"
    torch_dtype: str = "float32"  # 改为float32以避免精度问题
    use_mps: bool = torch.backends.mps.is_available()
    use_cuda: bool = torch.cuda.is_available()
    
    # 输出配置
    output_dir: str = "./outputs"
    logging_dir: str = "./logs"
    
    # 其他配置
    seed: int = 42
    
    def get_device(self):
        """获取训练设备"""
        if self.use_mps and torch.backends.mps.is_available():
            return torch.device("mps")
        elif self.use_cuda and torch.cuda.is_available():
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