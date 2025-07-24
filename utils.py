import re
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

def extract_answer(text: str) -> str:
    """从文本中提取数学答案"""
    # 查找 #### 后面的数字
    match = re.search(r'####\s*(\d+(?:\.\d+)?)', text)
    if match:
        return match.group(1).strip()
    
    # 查找最后一个数字
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]
    
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

def load_math_dataset(config) -> Dataset:
    """加载数学数据集"""
    print("正在加载GSM8K数据集...")
    dataset = load_dataset(config.dataset_name, config.dataset_config, split=config.dataset_split)
    
    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
    
    print(f"数据集加载完成，共{len(dataset)}条样本")
    return dataset

def format_math_prompt(question: str) -> str:
    """格式化数学问题prompt"""
    return f"""请解决下面的数学问题，逐步思考并给出最终答案。

问题: {question}

解答: """

def prepare_dataset_for_grpo(dataset: Dataset, tokenizer: AutoTokenizer, config) -> Dataset:
    """为GRPO训练准备数据集"""
    
    def tokenize_function(examples):
        prompts = [format_math_prompt(q) for q in examples['question']]
        
        # 对问题进行tokenize
        model_inputs = tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
            return_tensors="pt"
        )
        
        # 保存ground truth答案
        model_inputs['ground_truth'] = [extract_answer(answer) for answer in examples['answer']]
        model_inputs['question'] = examples['question']
        model_inputs['full_answer'] = examples['answer']
        
        return model_inputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def count_tokens_in_response(text: str, tokenizer: AutoTokenizer) -> int:
    """统计响应中的token数量"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def generate_response(model, tokenizer, prompt: str, config, max_new_tokens: int = 200) -> Tuple[str, int]:
    """生成模型响应"""
    device = config.get_device()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码响应
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response[len(prompt):].strip()
    
    # 统计token数量
    token_count = count_tokens_in_response(response, tokenizer)
    
    return response, token_count

def save_metrics(metrics: Dict[str, List[float]], output_dir: str):
    """保存训练指标"""
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"训练指标已保存到: {metrics_path}")

def setup_model_and_tokenizer(config):
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