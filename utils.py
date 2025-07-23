import re
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from typing import List, Dict, Tuple, Optional


def load_gsm8k_dataset(split: str = "train", max_samples: Optional[int] = None):
    """加载GSM8K数据集"""
    dataset = load_dataset("gsm8k", "main")[split]
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def extract_answer(text: str) -> Optional[float]:
    """从文本中提取数值答案"""
    # 寻找 #### 后面的数字
    if "####" in text:
        answer_part = text.split("####")[-1].strip()
    else:
        answer_part = text.strip()
    
    # 移除货币符号和逗号
    answer_part = re.sub(r'[\$,]', '', answer_part)
    
    # 提取数字
    numbers = re.findall(r'-?\d+\.?\d*', answer_part)
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            pass
    return None


def compute_reward(predicted_answer: str, true_answer: str) -> float:
    """计算奖励（简单版本：基于答案正确性）"""
    pred_num = extract_answer(predicted_answer)
    true_num = extract_answer(true_answer)
    
    if pred_num is None or true_num is None:
        return 0.0
    
    # 允许小的数值误差
    if abs(pred_num - true_num) < 1e-6:
        return 1.0
    else:
        return 0.0


def prepare_prompt(question: str) -> str:
    """准备输入提示"""
    return f"解决这个数学问题，并在最后一行用 #### 给出数值答案：\n\n{question}\n\n"


def count_tokens(text: str, tokenizer) -> int:
    """计算文本的token数量"""
    return len(tokenizer.encode(text, add_special_tokens=False))


def batch_process_data(examples: List[Dict], tokenizer, max_length: int = 512) -> Dict:
    """批量处理数据"""
    prompts = [prepare_prompt(ex["question"]) for ex in examples]
    
    # 编码prompts
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "questions": [ex["question"] for ex in examples],
        "answers": [ex["answer"] for ex in examples]
    }


def log_ratio_clip(log_ratio: torch.Tensor, clip_ratio: float = 0.2) -> torch.Tensor:
    """剪切对数比率以避免过大的策略更新"""
    return torch.clamp(log_ratio, -clip_ratio, clip_ratio)


def compute_advantages(rewards: torch.Tensor, values: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """计算优势函数（简化版本）"""
    advantages = rewards - values
    return advantages


def save_metrics(metrics: Dict, save_path: str):
    """保存训练指标"""
    import json
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def load_metrics(load_path: str) -> Dict:
    """加载训练指标"""
    import json
    try:
        with open(load_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "rewards": [],
            "losses": [],
            "token_counts": [],
            "epochs": []
        } 