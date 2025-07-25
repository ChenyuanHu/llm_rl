import re
import torch
import numpy as np
from typing import List, Dict, Tuple
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from config import TrainingConfig

def extract_answer(text: str) -> str:
    """从文本中提取数学答案"""
    # 查找 #### 后面的数字
    match = re.findall(r'####\s*(\d+(?:\.\d+)?)', text)
    if match:
        return match[-1].strip()
    
    # 查找最后一个数字
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].strip()
    
    return ""

def extract_predicted_answer(text: str) -> str:
    """从文本中提取预测的数学答案"""
    # 查找 \boxed{3500} 中的数字
    match = re.findall(r'\\boxed\{(\d+(?:\.\d+)?)\}', text)
    if match:
        return match[-1].strip()
    
    return ""

def is_correct_answer(predicted: str, ground_truth: str) -> bool:
    """检查答案是否正确"""
    try:
        pred_num = float(predicted.replace(',', ''))
        gt_num = float(ground_truth.replace(',', ''))
        return abs(pred_num - gt_num) < 1e-6
    except (ValueError, TypeError):
        return predicted.strip().lower() == ground_truth.strip().lower()

def compute_reward(predicted_answer: str, ground_truth: str) -> float:
    """计算reward分数"""
    if is_correct_answer(predicted_answer, ground_truth):
        return 1.0
    else:
        return -0.1  # 轻微惩罚错误答案

def load_math_dataset(config: TrainingConfig) -> Dataset:
    """加载数学数据集"""
    print("正在加载GSM8K数据集...")
    dataset = load_dataset(config.dataset_name, config.dataset_config, split=config.dataset_split)
    
    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
    
    print(f"数据集加载完成，共{len(dataset)}条样本")
    return dataset

def format_math_chat_input(question: str, tokenizer: AutoTokenizer) -> str:
        # 使用Qwen的对话格式
        messages = [
            {"role": "system", "content": "你是一个专业的数学助手，擅长解决各种数学问题。请逐步思考并给出准确答案。最终答案放在最后并放在\\boxed{}中，如\\boxed{最终答案}。"},
            {"role": "user", "content": question}
        ]
        
        # 尝试使用chat template
        chat_input = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return chat_input

def save_metrics(metrics: Dict[str, List[float]], output_dir: str):
    """保存训练指标"""
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"训练指标已保存到: {metrics_path}")

def setup_model_and_tokenizer(config: TrainingConfig):
    """设置模型和tokenizer"""
    print(f"正在加载模型: {config.model_name}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        padding_side="left"
    )
    
    # 设置pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=config.get_torch_dtype(),
        device_map=config.device_map
    )
    
    print("模型和tokenizer加载完成")
    return model, tokenizer

class MetricsTracker:
    """训练指标跟踪器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.token_counts = []
        self.rewards = []
        self.losses = []
        self.steps = []
    
    def add_metrics(self, step: int, token_count: float, reward: float, loss: float):
        self.steps.append(step)
        self.token_counts.append(token_count)
        self.rewards.append(reward)
        self.losses.append(loss)
    
    def get_averages(self) -> Dict[str, float]:
        if not self.token_counts:
            return {"avg_tokens": 0, "avg_reward": 0, "avg_loss": 0}
        
        return {
            "avg_tokens": np.mean(self.token_counts),
            "avg_reward": np.mean(self.rewards),
            "avg_loss": np.mean(self.losses)
        }
    
    def save_to_dict(self) -> Dict[str, List]:
        return {
            "steps": self.steps,
            "token_counts": self.token_counts,
            "rewards": self.rewards,
            "losses": self.losses
        } 