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
        
    def generate_and_evaluate(self, question: str, ground_truth: str) -> Dict:
        """生成响应并评估 - 合并相关操作"""
        # 准备输入
        prompt = format_math_chat_input(question, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt", 
                               truncation=True, max_length=self.config.max_length).to(self.device)
        
        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 提取和解码响应
        input_length = inputs['input_ids'].shape[1]
        response_ids = outputs[0][input_length:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # 评估响应
        predicted_answer = extract_predicted_answer(response)
        reward = compute_reward(predicted_answer, ground_truth)
        
        return {
            'prompt': prompt,
            'response': response,
            'token_count': len(response_ids),
            'input_ids': inputs['input_ids'],
            'response_ids': response_ids,
            'predicted_answer': predicted_answer,
            'ground_truth': ground_truth,
            'reward': reward
        }
    
    def compute_policy_loss(self, input_ids, response_ids, reward):
        """计算策略损失 - 简化版本"""
        if len(response_ids) == 0:
            return None, 0.0
            
        # 确保形状正确
        if response_ids.dim() == 1:
            response_ids = response_ids.unsqueeze(0)
        
        # 前向传播
        full_ids = torch.cat([input_ids, response_ids], dim=1)
        outputs = self.model(full_ids)
        
        # 计算损失
        prompt_length = input_ids.shape[1]
        response_length = response_ids.shape[1]
        
        # 提取响应部分的logits并计算概率
        response_logits = outputs.logits[:, prompt_length-1:prompt_length+response_length-1]
        log_probs = F.log_softmax(response_logits, dim=-1)
        selected_log_probs = log_probs[0].gather(1, response_ids[0].unsqueeze(1)).squeeze()
        
        # 策略损失
        current_log_prob = selected_log_probs.mean()
        policy_loss = -current_log_prob * reward
        
        return policy_loss, current_log_prob.item()
    
    def train_step(self, batch_data) -> Dict:
        """执行一个训练步骤 - 优化版本"""
        total_loss = 0
        total_reward = 0
        total_tokens = 0
        valid_samples = 0

        for item in batch_data:
            # 生成和评估
            result = self.generate_and_evaluate(
                item['question'], 
                extract_answer(item['answer'])
            )

            if result['reward'] < 0:
                print(f"prompt: {result['prompt']}")
                print(f"response: {result['response']}")

            print(f"predicted_answer: {result['predicted_answer']}")
            print(f"ground_truth: {result['ground_truth']}")
            print(f"reward: {result['reward']}")
            print(f"token_count: {result['token_count']}")
            
            # 计算损失并更新
            loss, log_prob = self.compute_policy_loss(
                result['input_ids'],
                result['response_ids'], 
                result['reward']
            )
            
            if loss is not None:
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                valid_samples += 1
            
            total_reward += result['reward']
            total_tokens += result['token_count']
        
        # 返回平均统计
        batch_size = len(batch_data)
        return {
            'avg_loss': total_loss / max(valid_samples, 1),
            'avg_reward': total_reward / batch_size,
            'avg_tokens': total_tokens / batch_size
        }
        
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
            step_stats = self.train_step(batch)
            cost_time = time.time() - start_time
            
            # 记录指标
            current_step = epoch * len(dataloader) + batch_idx
            self.metrics_tracker.add_metrics(
                current_step, step_stats['avg_tokens'], 
                step_stats['avg_reward'], step_stats['avg_loss']
            )
            
            for key in epoch_metrics:
                epoch_metrics[key].append(step_stats[f"avg_{key[:-1]}"])
            
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