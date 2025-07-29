import os
import time
import torch
import numpy as np
from typing import Dict, List
from transformers import set_seed
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import TrainingConfig
from utils import (
    load_math_dataset, extract_answer, extract_predicted_answer,
    compute_reward, save_metrics, setup_model_and_tokenizer,
    MetricsTracker, format_math_chat_input
)

class TrainingUtils:
    """训练工具类 - 分离工具方法"""
    
    @staticmethod
    def create_dataloader(dataset, config):
        """创建数据加载器"""
        batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: batch
        ), batch_size

    @staticmethod
    def log_training_step(epoch, total_epochs, batch_idx, total_batches, 
                         stats, cost_time):
        """统一的训练日志输出"""
        print(f"Epoch {epoch}/{total_epochs}, "
              f"Batch {batch_idx}/{total_batches}, "
              f"Tokens: {stats['avg_tokens']:.1f}, "
              f"Reward: {stats['avg_reward']:.3f}, "
              f"Loss: {stats['avg_loss']:.4f}, "
              f"LogProb: {stats.get('avg_log_prob', 0):.3f}, "
              f"Time: {cost_time:.2f}s")

    @staticmethod
    def save_checkpoint(model, tokenizer, epoch, output_dir):
        """保存检查点"""
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        return checkpoint_dir

class SimpleGRPOTrainer:
    """简化的GRPO训练器 - 优化版"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.get_device()
        self.metrics_tracker = MetricsTracker()
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        
        # 初始化
        set_seed(config.seed)
        self._print_device_info()
        
    def _print_device_info(self):
        """打印设备信息"""
        print(f"使用设备: {self.device}")
        print(f"PyTorch版本: {torch.__version__}")
        if self.config.use_mps:
            print("MPS可用，将使用Apple Silicon GPU加速")
        
    def setup(self):
        """统一的设置方法"""
        print("正在设置模型和数据...")
        
        # 设置模型
        self.model, self.tokenizer = setup_model_and_tokenizer(self.config)
        self.model.to(self.device)

        self.ref_model, _ = setup_model_and_tokenizer(self.config, load_tokenizer=False)
        self.ref_model.to(self.device)
        
        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        
        # 设置优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # 设置数据集
        self.dataset = load_math_dataset(self.config)
        
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.logging_dir, exist_ok=True)
        
        print(f"设置完成 - 模型已加载，数据集: {len(self.dataset)}条样本")
        
    def generate_and_evaluate_batch(self, batch_data: List[Dict]) -> List[Dict]:
        """批量生成响应并评估 - 优化版本"""
        # 准备批量输入
        prompts = []
        questions = []
        ground_truths = []
        
        for item in batch_data:
            question = item['question']
            ground_truth = extract_answer(item['answer'])
            prompt = format_math_chat_input(question, self.tokenizer)
            
            prompts.append(prompt)
            questions.append(question)
            ground_truths.append(ground_truth)
        
        # 批量tokenize
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True, 
            max_length=self.config.max_length
        ).to(self.device)
        
        # 批量生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=0.5,
                # top_k=10,
                # top_p=0.9,
                # repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 处理批量结果
        results = []
        input_length = inputs['input_ids'].shape[1]
        
        for i in range(len(batch_data)):
            # 提取响应部分
            response_ids = outputs[i][input_length:]
            
            # 安全的EOS token查找 - 使用torch操作而非Python list
            eos_mask = (response_ids == self.tokenizer.eos_token_id)
            if eos_mask.any():
                # 找到第一个EOS token的位置
                eos_positions = torch.nonzero(eos_mask, as_tuple=False)
                if len(eos_positions) > 0:
                    last_eos_index = eos_positions[0].item()
                    response_ids = response_ids[:last_eos_index]
            # 如果没有EOS token，保持原始长度（这是正常情况）
            
            # 解码响应
            response = self.tokenizer.decode(response_ids, skip_special_tokens=False)
            
            # 评估响应
            predicted_answer = extract_predicted_answer(response)
            # 优化奖励计算 - 使用tensor长度而非Python len()
            # token_penalty = 0.005 * response_ids.shape[0]  # 过长思考惩罚
            token_penalty = 0
            reward = compute_reward(predicted_answer, ground_truths[i]) - token_penalty
            
            results.append({
                'prompt': prompts[i],
                'response': response,
                'token_count': response_ids.shape[0],  # 使用tensor.shape[0]而非len()
                'input_ids': inputs['input_ids'][i:i+1],  # 保持batch维度
                'response_ids': response_ids,
                'predicted_answer': predicted_answer,
                'ground_truth': ground_truths[i],
                'reward': reward
            })
            
        return results
    
    def compute_grpo_loss(self, batch_results: List[Dict]):
        """完整的GRPO损失计算 - 完全对齐原始论文实现
        
        GRPO (Group Relative Policy Optimization) 原始论文公式：
        L = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)] - β * KL(π_θ || π_ref)
        
        其中：
        - r(θ) = π_θ(y|x) / π_ref(y|x) 是重要性采样比率
        - A 是相对优势 (group内z-score标准化)
        - clip() 是PPO风格的clipping
        - KL是对参考策略的KL散度约束
        """
        if len(batch_results) == 0:
            return None, 0.0
            
        # 准备数据
        all_input_ids = []
        all_response_ids = []
        all_rewards = []
        
        for result in batch_results:
            if len(result['response_ids']) > 0:
                all_input_ids.append(result['input_ids'].squeeze(0))
                all_response_ids.append(result['response_ids'])
                all_rewards.append(result['reward'])
        
        if len(all_rewards) == 0:
            return None, 0.0
            
        # 转换为tensor
        rewards = torch.tensor(all_rewards, dtype=torch.float32, device=self.device)

        # GRPO核心：计算group内的relative advantage (z-score标准化)
        if len(rewards) > 1:
            reward_mean = rewards.mean()
            reward_std = rewards.std() + 1e-8  # 避免除零
            advantages = (rewards - reward_mean) / reward_std
        else:
            advantages = rewards
        
        # 高效批处理：准备批量数据
        batch_full_ids = []
        batch_prompt_lengths = []
        batch_response_lengths = []
        
        for input_ids, response_ids in zip(all_input_ids, all_response_ids):
            if response_ids.dim() == 1:
                response_ids = response_ids.unsqueeze(0)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                
            full_ids = torch.cat([input_ids, response_ids], dim=1)
            batch_full_ids.append(full_ids.squeeze(0))
            batch_prompt_lengths.append(input_ids.shape[1])
            batch_response_lengths.append(response_ids.shape[1])
        
        # 序列padding和批处理
        max_length = max(seq.shape[0] for seq in batch_full_ids)
        padded_batch = torch.full(
            (len(batch_full_ids), max_length), 
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            dtype=torch.long,
            device=self.device
        )
        
        attention_mask = torch.zeros_like(padded_batch)
        
        for i, seq in enumerate(batch_full_ids):
            seq_len = seq.shape[0]
            padded_batch[i, :seq_len] = seq
            attention_mask[i, :seq_len] = 1
        
        # 当前策略前向传播（保留梯度）
        current_outputs = self.model(padded_batch, attention_mask=attention_mask)
        
        # 参考策略前向传播（无梯度）
        with torch.no_grad():
            ref_outputs = self.ref_model(padded_batch, attention_mask=attention_mask)
        
        # GRPO损失计算
        total_policy_loss = 0
        total_kl_loss = 0
        total_log_prob = 0
        valid_samples = 0
        
        # GRPO超参数
        clip_epsilon = self.config.clip_epsilon
        kl_coeff = self.config.kl_coeff
        
        for i, (prompt_length, response_length, advantage) in enumerate(
            zip(batch_prompt_lengths, batch_response_lengths, advantages)
        ):
            # 提取response部分的logits
            current_logits = current_outputs.logits[i]
            ref_logits = ref_outputs.logits[i]
            
            response_start = prompt_length - 1  # autoregressive shift
            response_end = prompt_length + response_length - 1
            
            current_response_logits = current_logits[response_start:response_end]
            ref_response_logits = ref_logits[response_start:response_end]
            
            # 提取response token ids
            response_targets = padded_batch[i][prompt_length:prompt_length + response_length]
            
            # 计算当前策略的log probabilities
            current_log_probs = F.log_softmax(current_response_logits, dim=-1)
            current_selected_log_probs = current_log_probs.gather(1, response_targets.unsqueeze(1)).squeeze()
            current_log_prob_sum = current_selected_log_probs.sum()
            
            # 计算参考策略的log probabilities
            ref_log_probs = F.log_softmax(ref_response_logits, dim=-1)
            ref_selected_log_probs = ref_log_probs.gather(1, response_targets.unsqueeze(1)).squeeze()
            ref_log_prob_sum = ref_selected_log_probs.sum()
            
            # 计算重要性采样比率 r(θ) = π_θ(y|x) / π_ref(y|x)
            log_ratio = current_log_prob_sum - ref_log_prob_sum
            ratio = torch.exp(log_ratio)
            
            # PPO/GRPO clipping
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
            
            # 计算clipped surrogate objective
            # L_clip = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]
            surrogate1 = ratio * advantage
            surrogate2 = clipped_ratio * advantage
            policy_loss = -torch.min(surrogate1, surrogate2)  # 负号因为我们要最大化
            
            # 计算KL散度
            # KL(π_θ || π_ref) 在response tokens上  
            # PyTorch KL散度: F.kl_div(log_probs, target_probs)
            ref_probs = F.softmax(ref_response_logits, dim=-1)
            kl_divergence = F.kl_div(current_log_probs, ref_probs, reduction='sum')
            
            total_policy_loss += policy_loss
            total_kl_loss += kl_divergence
            total_log_prob += current_log_prob_sum.item()
            valid_samples += 1
        
        if valid_samples == 0:
            return None, 0.0
        
        # 计算最终的GRPO损失
        # L_total = L_clip - β * KL(π_θ || π_ref)
        avg_policy_loss = total_policy_loss / valid_samples
        avg_kl_loss = total_kl_loss / valid_samples
        
        # 原始GRPO论文的完整损失公式
        total_loss = avg_policy_loss + kl_coeff * avg_kl_loss
        
        avg_log_prob = total_log_prob / valid_samples
        
        # 详细调试信息
        if len(rewards) > 1:
            print(f"  GRPO Complete Stats:")
            print(f"    Rewards: {rewards}, [{rewards.min():.3f}, {rewards.max():.3f}], Mean: {rewards.mean():.3f}, Std: {rewards.std():.3f}")
            print(f"    Advantages: {advantages}, [{advantages.min():.3f}, {advantages.max():.3f}]")
            print(f"    Policy Loss: {avg_policy_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")
            print(f"    Total Loss: {total_loss:.4f}, Log Prob: {avg_log_prob:.3f}")
        
        return total_loss, avg_log_prob
    
    def train_group(self, group_data) -> Dict:
        """执行一个训练步骤 - 批处理GPU优化版本"""
        # 批量生成和评估
        batch_results = self.generate_and_evaluate_batch(group_data)
        print(f"batch_results: {batch_results}")
        
        # 计算统计信息
        total_reward = sum(result['reward'] for result in batch_results)
        total_tokens = sum(result['token_count'] for result in batch_results)
        batch_size = len(batch_results)
        
        # 打印详细信息（抽样显示）
        negative_count = sum(1 for r in batch_results if r['reward'] < 0)
        sample_displayed = 0
        max_display = 0  # 最多显示3个样本的详细信息
        
        for i, result in enumerate(batch_results):
            should_display = sample_displayed < max_display
            if should_display:
                print(f"Sample {i+1}/{batch_size}:")
                print(f"  predicted_answer: {result['predicted_answer']}")
                print(f"  ground_truth: {result['ground_truth']}")
                print(f"  reward: {result['reward']}")
                print(f"  token_count: {result['token_count']}")
                if result['reward'] < 0:
                    print(f"  prompt: {result['prompt'][:100]}...")
                    print(f"  response: {result['response'][:100]}...")
                sample_displayed += 1
        
        print(f"负奖励样本数: {negative_count}/{batch_size}")
        
        # 使用GRPO计算损失
        loss, avg_log_prob = self.compute_grpo_loss(batch_results)
        
        if loss is not None:
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            loss_value = loss.item()
        else:
            loss_value = 0.0
        
        # 返回平均统计
        return {
            'avg_loss': loss_value,
            'avg_reward': total_reward / batch_size,
            'avg_tokens': total_tokens / batch_size,
            'avg_log_prob': avg_log_prob if loss is not None else 0.0
        }

    def sample_group(self, batch_data) -> List[Dict]:
        """GRPO group采样 - 为单个prompt创建group_size个副本供生成多个responses"""
        assert len(batch_data) == 1
        
        # 为同一个prompt创建group_size个副本
        # generate_and_evaluate_batch会为每个副本生成不同的response
        return [batch_data[0]] * self.config.group_size
        
    def train_epoch(self, epoch: int) -> Dict:
        """训练一个epoch - 简化版本"""
        print(f"\n开始训练 Epoch {epoch + 1}/{self.config.num_train_epochs}")
        
        self.model.train()
        
        # 创建数据加载器
        dataloader, batch_size = TrainingUtils.create_dataloader(self.dataset, self.config)
        print(f"批处理大小: {batch_size}, 批次数: {len(dataloader)}")
        
        # 训练指标
        epoch_metrics = {'token_counts': [], 'rewards': [], 'losses': []}
        
        for batch_idx, batch in enumerate(dataloader):
            # 执行训练步骤
            start_time = time.time()
            group = self.sample_group(batch)
            step_stats = self.train_group(group)
            cost_time = time.time() - start_time
            
            # 记录指标
            current_step = epoch * len(dataloader) + batch_idx
            self.metrics_tracker.add_metrics(
                current_step, step_stats['avg_tokens'], 
                step_stats['avg_reward'], step_stats['avg_loss']
            )
            
            epoch_metrics['token_counts'].append(step_stats['avg_tokens'])
            epoch_metrics['rewards'].append(step_stats['avg_reward'])
            epoch_metrics['losses'].append(step_stats['avg_loss'])
            if 'avg_log_prob' in step_stats:
                if 'log_probs' not in epoch_metrics:
                    epoch_metrics['log_probs'] = []
                epoch_metrics['log_probs'].append(step_stats['avg_log_prob'])
            
            # 日志输出
            if (batch_idx + 1) % self.config.logging_steps == 0:
                TrainingUtils.log_training_step(
                    epoch + 1, self.config.num_train_epochs,
                    batch_idx + 1, len(dataloader),
                    step_stats, cost_time
                )
        
        return {f"avg_{k[:-1]}": np.mean(v) for k, v in epoch_metrics.items()}
        
    def train(self):
        """主训练循环 - 简化版本"""
        print("="*60)
        print("开始GRPO训练")
        print("="*60)
        
        # 统一设置
        self.setup()
        
        # 训练循环
        for epoch in range(self.config.num_train_epochs):
            epoch_stats = self.train_epoch(epoch)
            
            # 输出epoch统计
            print(f"\nEpoch {epoch + 1} 完成:")
            for key, value in epoch_stats.items():
                print(f"  {key}: {value:.3f}")
            
            # 保存检查点
            checkpoint_dir = TrainingUtils.save_checkpoint(
                self.model, self.tokenizer, epoch + 1, self.config.output_dir
            )
            print(f"检查点已保存: {checkpoint_dir}")
        
        # 保存最终结果
        self._save_final_results()
        print("训练完成！")
        
    def _save_final_results(self):
        """保存最终结果"""
        # 保存最终模型
        final_model_dir = os.path.join(self.config.output_dir, "final_model")
        self.model.save_pretrained(final_model_dir)
        self.tokenizer.save_pretrained(final_model_dir)
        
        # 保存训练指标
        metrics_dict = self.metrics_tracker.save_to_dict()
        save_metrics(metrics_dict, self.config.output_dir)
        
        print(f"最终模型保存至: {final_model_dir}")

def main():
    """主函数 - 简化版本"""
    print("🚀 初始化GRPO训练...")
    
    try:
        # 创建训练器并训练
        config = TrainingConfig()
        trainer = SimpleGRPOTrainer(config)
        trainer.train()
        
        # 训练总结
        print("\n📊 训练总结:")
        final_stats = trainer.metrics_tracker.get_averages()
        for key, value in final_stats.items():
            print(f"  {key}: {value:.3f}")
        
        print("\n💡 运行可视化: python visualize.py")
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        raise

if __name__ == "__main__":
    main() 