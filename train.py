import os
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List
import wandb
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed
)
from trl import (
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    create_reference_model
)
from trl.core import LengthSampler

from config import TrainingConfig
from utils import (
    load_math_dataset,
    prepare_dataset_for_grpo,
    extract_answer,
    compute_reward,
    generate_response,
    save_metrics,
    setup_model_and_tokenizer,
    MetricsTracker,
    format_math_prompt
)

class GRPOTrainer:
    """GRPO训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.get_device()
        self.metrics_tracker = MetricsTracker()
        
        # 设置随机种子
        set_seed(config.seed)
        
        # 初始化wandb（可选）
        self.use_wandb = False  # 设为True启用wandb日志
        
        print(f"使用设备: {self.device}")
        print(f"PyTorch版本: {torch.__version__}")
        if config.use_mps:
            print("MPS可用，将使用Apple Silicon GPU加速")
        
    def setup_models(self):
        """设置模型和tokenizer"""
        # 加载基础模型和tokenizer
        self.model, self.tokenizer = setup_model_and_tokenizer(self.config)
        
        # 创建带有价值头的模型用于PPO训练
        self.ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=self.config.get_torch_dtype(),
            device_map=self.config.device_map
        )
        
        # 创建参考模型
        self.ref_model = create_reference_model(self.ppo_model)
        
        print("GRPO模型设置完成")
        
    def setup_dataset(self):
        """设置训练数据集"""
        # 加载原始数据集
        raw_dataset = load_math_dataset(self.config)
        
        # 准备数据集
        self.dataset = prepare_dataset_for_grpo(raw_dataset, self.tokenizer, self.config)
        
        print(f"数据集准备完成，训练样本数: {len(self.dataset)}")
        
    def setup_ppo_config(self):
        """设置PPO配置"""
        # 使用最基本的PPOConfig参数，明确禁用bf16
        self.ppo_config = PPOConfig(
            learning_rate=self.config.learning_rate,
            batch_size=self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps,
            mini_batch_size=self.config.per_device_train_batch_size,
            bf16=False,  # 明确禁用bf16
            fp16=False   # 明确禁用fp16
        )
        
    def setup_ppo_trainer(self):
        """设置PPO训练器"""
        self.ppo_trainer = PPOTrainer(
            self.ppo_config,
            self.ppo_model,
            self.ref_model,
            self.tokenizer,
        )
        
    def generate_responses(self, queries: List[str]) -> List[str]:
        """生成模型响应"""
        responses = []
        
        for query in queries:
            # 使用模型生成响应
            query_tensor = self.tokenizer.encode(query, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                response_tensor = self.ppo_trainer.generate(
                    query_tensor,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码响应（只取新生成的部分）
            response = self.tokenizer.decode(
                response_tensor[0][len(query_tensor[0]):], 
                skip_special_tokens=True
            ).strip()
            
            responses.append(response)
            
        return responses
        
    def compute_rewards(self, responses: List[str], ground_truths: List[str]) -> List[float]:
        """计算reward分数"""
        rewards = []
        
        for response, gt in zip(responses, ground_truths):
            # 从响应中提取答案
            predicted_answer = extract_answer(response)
            
            # 计算reward
            reward = compute_reward(predicted_answer, gt)
            rewards.append(reward)
            
        return rewards
        
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        print(f"\n开始训练 Epoch {epoch + 1}/{self.config.num_train_epochs}")
        
        # 准备batch数据
        batch_size = self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps
        num_batches = len(self.dataset) // batch_size
        
        epoch_metrics = {
            'token_counts': [],
            'rewards': [],
            'losses': []
        }
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(self.dataset))
            batch = self.dataset.select(range(start_idx, end_idx))
            
            # 准备查询和ground truth
            queries = [format_math_prompt(item['question']) for item in batch]
            ground_truths = [item['ground_truth'] for item in batch]
            
            # 生成响应
            responses = self.generate_responses(queries)
            
            # 计算rewards
            rewards = self.compute_rewards(responses, ground_truths)
            
            # 计算token数量
            token_counts = [len(self.tokenizer.encode(resp, add_special_tokens=False)) for resp in responses]
            avg_token_count = np.mean(token_counts)
            
            # 准备PPO训练数据
            query_tensors = [self.tokenizer.encode(q, return_tensors="pt")[0] for q in queries]
            response_tensors = [self.tokenizer.encode(r, return_tensors="pt")[0] for r in responses]
            reward_tensors = [torch.tensor(r, dtype=torch.float32) for r in rewards]
            
            # PPO训练步骤
            stats = self.ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
            
            # 记录指标
            current_step = epoch * num_batches + batch_idx
            avg_reward = np.mean(rewards)
            avg_loss = stats.get('ppo/loss/total', 0.0)
            
            self.metrics_tracker.add_metrics(current_step, avg_token_count, avg_reward, avg_loss)
            
            epoch_metrics['token_counts'].append(avg_token_count)
            epoch_metrics['rewards'].append(avg_reward)
            epoch_metrics['losses'].append(avg_loss)
            
            # 日志输出
            if (batch_idx + 1) % self.config.logging_steps == 0:
                print(f"Epoch {epoch + 1}/{self.config.num_train_epochs}, "
                      f"Batch {batch_idx + 1}/{num_batches}, "
                      f"Avg Tokens: {avg_token_count:.1f}, "
                      f"Avg Reward: {avg_reward:.3f}, "
                      f"Loss: {avg_loss:.4f}")
                
                # 显示一个样例
                if len(responses) > 0:
                    print(f"样例问题: {batch[0]['question'][:100]}...")
                    print(f"模型回答: {responses[0][:150]}...")
                    print(f"正确答案: {ground_truths[0]}")
                    print(f"Reward: {rewards[0]:.3f}")
                    print("-" * 50)
        
        # 返回epoch统计
        return {
            'avg_token_count': np.mean(epoch_metrics['token_counts']),
            'avg_reward': np.mean(epoch_metrics['rewards']),
            'avg_loss': np.mean(epoch_metrics['losses'])
        }
        
    def train(self):
        """主训练循环"""
        print("="*60)
        print("开始GRPO训练")
        print("="*60)
        
        # 设置模型和数据
        self.setup_models()
        self.setup_dataset()
        self.setup_ppo_config()
        self.setup_ppo_trainer()
        
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.logging_dir, exist_ok=True)
        
        # 训练循环
        for epoch in range(self.config.num_train_epochs):
            epoch_stats = self.train_epoch(epoch)
            
            print(f"\nEpoch {epoch + 1} 完成:")
            print(f"  平均Token数: {epoch_stats['avg_token_count']:.1f}")
            print(f"  平均Reward: {epoch_stats['avg_reward']:.3f}")
            print(f"  平均Loss: {epoch_stats['avg_loss']:.4f}")
            
            # 保存检查点
            if (epoch + 1) % 1 == 0:  # 每个epoch保存一次
                checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-epoch-{epoch+1}")
                self.ppo_trainer.save_pretrained(checkpoint_dir)
                print(f"检查点已保存到: {checkpoint_dir}")
        
        # 保存最终模型
        final_model_dir = os.path.join(self.config.output_dir, "final_model")
        self.ppo_trainer.save_pretrained(final_model_dir)
        print(f"最终模型已保存到: {final_model_dir}")
        
        # 保存训练指标
        metrics_dict = self.metrics_tracker.save_to_dict()
        save_metrics(metrics_dict, self.config.output_dir)
        
        print("训练完成！")
        return metrics_dict

def main():
    """主函数"""
    print("初始化GRPO训练...")
    
    # 创建配置
    config = TrainingConfig()
    
    # 创建训练器
    trainer = GRPOTrainer(config)
    
    # 开始训练
    metrics = trainer.train()
    
    print("\n训练总结:")
    final_stats = trainer.metrics_tracker.get_averages()
    print(f"平均Token数: {final_stats['avg_tokens']:.1f}")
    print(f"平均Reward: {final_stats['avg_reward']:.3f}")
    print(f"平均Loss: {final_stats['avg_loss']:.4f}")
    
    # 提示运行可视化
    print("\n运行以下命令查看训练可视化:")
    print("python visualize.py")

if __name__ == "__main__":
    main() 