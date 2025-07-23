import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os
import json
from typing import List, Dict

from utils import (
    load_gsm8k_dataset, 
    compute_reward, 
    batch_process_data,
    count_tokens,
    save_metrics,
    load_metrics,
    prepare_prompt,
    extract_answer
)
from grpo import GRPOTrainer
from visualize import plot_training_metrics


def main():
    # 配置参数
    config = {
        "model_name": "Qwen/Qwen2.5-0.5B",  # 使用0.5B版本，更接近0.6B需求
        "max_samples_train": 50,   # 先用少量样本测试
        "max_samples_eval": 20,    # 评估样本数量
        "batch_size": 2,           # 减少批次大小
        "num_epochs": 3,
        "learning_rate": 1e-5,
        "max_new_tokens": 128,     # 减少最大token数
        "max_length": 256,         # 减少最大长度
        "group_size": 2,           # 减少组大小
        "save_dir": "./outputs",
        "device": "cpu"  # 先用CPU避免MPS兼容性问题
    }
    
    print(f"使用设备: {config['device']}")
    
    # 创建输出目录
    os.makedirs(config["save_dir"], exist_ok=True)
    
    # 加载模型和tokenizer
    print("加载模型和tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 设置左对齐用于生成任务
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.float32,  # 统一使用float32
        device_map=config["device"]
    )
    
    # 加载数据集
    print("加载GSM8K数据集...")
    train_dataset = load_gsm8k_dataset("train", config["max_samples_train"])
    eval_dataset = load_gsm8k_dataset("test", config["max_samples_eval"])
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(eval_dataset)}")
    
    # 初始化GRPO训练器
    print("初始化GRPO训练器...")
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        learning_rate=config["learning_rate"],
        group_size=config["group_size"]
    )
    
    # 训练指标
    metrics = load_metrics(os.path.join(config["save_dir"], "metrics.json"))
    
    # 训练循环
    print("开始训练...")
    for epoch in range(config["num_epochs"]):
        print(f"\n=== Epoch {epoch + 1}/{config['num_epochs']} ===")
        
        epoch_losses = []
        epoch_rewards = []
        epoch_token_counts = []
        
        # 创建数据加载器
        train_data = [train_dataset[i] for i in range(len(train_dataset))]
        
        for i in tqdm(range(0, len(train_data), config["batch_size"]), desc="训练"):
            batch_data = train_data[i:i + config["batch_size"]]
            
            # 处理批次数据
            batch = batch_process_data(batch_data, tokenizer, config["max_length"])
            
            input_ids = batch["input_ids"].to(config["device"])
            attention_mask = batch["attention_mask"].to(config["device"])
            
            # 生成响应
            response_sequences, _ = trainer.generate_responses(
                input_ids, 
                attention_mask,
                max_new_tokens=config["max_new_tokens"]
            )
            
            # 计算奖励
            rewards = []
            token_counts = []
            
            for j, (response_seq, true_answer) in enumerate(zip(response_sequences, batch["answers"])):
                # 解码响应
                response_text = tokenizer.decode(
                    response_seq[len(input_ids[j]):], 
                    skip_special_tokens=True
                )
                
                # 计算奖励
                reward = compute_reward(response_text, true_answer)
                rewards.append(reward)
                
                # 计算token数量
                token_count = len(response_seq) - len(input_ids[j])
                token_counts.append(token_count)
            
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=config["device"])
            
            # 创建响应的attention mask
            response_mask = torch.ones_like(response_sequences, dtype=torch.float32, device=config["device"])
            response_mask[response_sequences == tokenizer.pad_token_id] = 0.0
            
            # 训练步骤
            step_metrics = trainer.train_step(
                input_ids=input_ids,
                attention_mask=attention_mask,
                response_ids=response_sequences,
                response_mask=response_mask,
                rewards=rewards_tensor
            )
            
            epoch_losses.append(step_metrics["total_loss"])
            epoch_rewards.append(step_metrics["mean_reward"])
            epoch_token_counts.extend(token_counts)
            
            # 每50步打印一次进度
            if (i // config["batch_size"] + 1) % 50 == 0:
                print(f"步骤 {i // config['batch_size'] + 1}: "
                      f"损失={step_metrics['total_loss']:.4f}, "
                      f"平均奖励={step_metrics['mean_reward']:.4f}")
        
        # 记录epoch指标
        avg_loss = np.mean(epoch_losses)
        avg_reward = np.mean(epoch_rewards)
        avg_token_count = np.mean(epoch_token_counts)
        
        metrics["epochs"].append(epoch + 1)
        metrics["losses"].append(avg_loss)
        metrics["rewards"].append(avg_reward)
        metrics["token_counts"].append(avg_token_count)
        
        print(f"Epoch {epoch + 1} 完成:")
        print(f"  平均损失: {avg_loss:.4f}")
        print(f"  平均奖励: {avg_reward:.4f}")
        print(f"  平均token数: {avg_token_count:.2f}")
        
        # 保存指标
        save_metrics(metrics, os.path.join(config["save_dir"], "metrics.json"))
        
        # 评估
        if (epoch + 1) % 1 == 0:  # 每个epoch评估一次
            eval_metrics = evaluate_model(trainer, eval_dataset, tokenizer, config)
            print(f"  评估准确率: {eval_metrics['accuracy']:.4f}")
            print(f"  评估平均token数: {eval_metrics['avg_tokens']:.2f}")
    
    # 保存最终模型
    model_save_path = os.path.join(config["save_dir"], "final_model")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"模型已保存到: {model_save_path}")
    
    # 绘制训练曲线
    print("生成训练曲线...")
    plot_training_metrics(metrics, config["save_dir"])
    
    print("训练完成！")


def evaluate_model(trainer, eval_dataset, tokenizer, config):
    """评估模型"""
    trainer.model.eval()
    
    correct = 0
    total = 0
    total_tokens = 0
    
    eval_data = [eval_dataset[i] for i in range(min(100, len(eval_dataset)))]  # 评估100个样本
    
    with torch.no_grad():
        for i in range(0, len(eval_data), config["batch_size"]):
            batch_data = eval_data[i:i + config["batch_size"]]
            batch = batch_process_data(batch_data, tokenizer, config["max_length"])
            
            input_ids = batch["input_ids"].to(config["device"])
            attention_mask = batch["attention_mask"].to(config["device"])
            
            # 生成响应
            response_sequences, _ = trainer.generate_responses(
                input_ids, 
                attention_mask,
                max_new_tokens=config["max_new_tokens"],
                temperature=0.1,  # 评估时使用更低的温度
                do_sample=True
            )
            
            for j, (response_seq, true_answer) in enumerate(zip(response_sequences, batch["answers"])):
                response_text = tokenizer.decode(
                    response_seq[len(input_ids[j]):], 
                    skip_special_tokens=True
                )
                
                reward = compute_reward(response_text, true_answer)
                if reward > 0.5:  # 正确答案
                    correct += 1
                
                token_count = len(response_seq) - len(input_ids[j])
                total_tokens += token_count
                total += 1
    
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "avg_tokens": total_tokens / total if total > 0 else 0.0
    }


if __name__ == "__main__":
    main() 