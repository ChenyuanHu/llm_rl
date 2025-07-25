import os
import time
import torch
import numpy as np
from typing import Dict
from transformers import set_seed
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import TrainingConfig
from utils import (
    load_math_dataset,
    extract_answer,
    compute_reward,
    save_metrics,
    setup_model_and_tokenizer,
    MetricsTracker,
    format_math_prompt,
    format_math_chat_input
)

class SimpleGRPOTrainer:
    """简化的GRPO训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.get_device()
        self.metrics_tracker = MetricsTracker()
        
        # 设置随机种子
        set_seed(config.seed)
        
        print(f"使用设备: {self.device}")
        print(f"PyTorch版本: {torch.__version__}")
        if config.use_mps:
            print("MPS可用，将使用Apple Silicon GPU加速")
        
    def setup_models(self):
        """设置模型和tokenizer"""
        # 加载基础模型和tokenizer
        self.model, self.tokenizer = setup_model_and_tokenizer(self.config)
        self.model.to(self.device)
        
        # 创建参考模型（用于KL散度计算）
        self.ref_model, _ = setup_model_and_tokenizer(self.config)
        self.ref_model.to(self.device)
        self.ref_model.eval()  # 参考模型保持eval模式
        
        # 设置优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        print("GRPO模型设置完成")
        
    def setup_dataset(self):
        """设置训练数据集"""
        # 加载原始数据集
        raw_dataset = load_math_dataset(self.config)
        self.dataset = raw_dataset
        print(f"数据集准备完成，训练样本数: {len(self.dataset)}")
        
    def generate_response(self, prompt: str) -> Dict:
        """生成模型响应并计算概率"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_length).to(self.device)
        
        with torch.no_grad():
            # 生成响应
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码响应
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = full_response[len(prompt):-len(self.tokenizer.eos_token)].strip()
        
        # 计算token数量
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
        token_count = len(response_tokens)
        
        return {
            'response': response,
            'token_count': token_count,
            'input_ids': inputs['input_ids'],
            'response_ids': outputs.sequences[0][len(inputs['input_ids'][0]):]
        }
    
    def compute_policy_loss(self, input_ids, response_ids, rewards, old_log_probs=None):
        """计算策略损失"""
        # 准备输入
        full_ids = torch.cat([input_ids, response_ids.unsqueeze(0)], dim=1)
        
        # 前向传播
        outputs = self.model(full_ids, labels=full_ids)
        logits = outputs.logits
        
        # 计算当前策略的log概率
        response_logits = logits[0, len(input_ids[0])-1:-1]  # 响应部分的logits
        log_probs = F.log_softmax(response_logits, dim=-1)
        
        # 选择实际生成的token的log概率
        selected_log_probs = log_probs.gather(1, response_ids.unsqueeze(1)).squeeze()
        current_log_prob = selected_log_probs.sum()
        
        # 简化的策略梯度损失
        policy_loss = -current_log_prob * rewards
        
        return policy_loss, current_log_prob
    
    def train_step(self, batch_data):
        """执行一个训练步骤"""
        total_loss = 0
        total_reward = 0
        total_tokens = 0
        
        for item in batch_data:
            question = item['question']
            ground_truth = extract_answer(item['answer'])
            prompt = format_math_prompt(question)
            
            # 生成响应
            gen_result = self.generate_response(prompt)
            response = gen_result['response']
            token_count = gen_result['token_count']
            
            # 计算奖励
            predicted_answer = extract_answer(response)
            reward = compute_reward(predicted_answer, ground_truth)
            
            # 计算损失
            if len(gen_result['response_ids']) > 0:
                loss, log_prob = self.compute_policy_loss(
                    gen_result['input_ids'],
                    gen_result['response_ids'],
                    reward
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            total_reward += reward
            total_tokens += token_count
        
        batch_size = len(batch_data)
        return {
            'avg_loss': total_loss / batch_size,
            'avg_reward': total_reward / batch_size,
            'avg_tokens': total_tokens / batch_size
        }
        
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        print(f"\n开始训练 Epoch {epoch + 1}/{self.config.num_train_epochs}")
        
        self.model.train()
        
        # 准备batch数据 - 使用datasets库的iter方法
        batch_size = self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps
        print(f"batch_size: {batch_size}")
        
        
        # 创建DataLoader
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,  # 每个epoch随机打乱
            collate_fn=lambda batch: batch  # 保持原始格式
        )
        
        num_batches = len(dataloader)
        print(f"num_batches: {num_batches}")
        
        epoch_metrics = {
            'token_counts': [],
            'rewards': [],
            'losses': []
        }
        
        for batch_idx, batch in enumerate(dataloader):
            sample_prompt = format_math_chat_input(batch[0]['question'], self.tokenizer)
            print(f"样例问题: {batch[0]['question']}")
            print(f"样例prompt: {sample_prompt}")
            sample_result = self.generate_response(sample_prompt)
            print(f"样例问题: {batch[0]['question']}")
            print(f"模型回答: {sample_result}")
            print(f"正确答案: {extract_answer(batch[0]['answer'])}")
            print("-" * 50)

            exit()
            
            # 执行训练步骤
            start_time = time.time()
            step_stats = self.train_step(batch)
            cost_time = time.time() - start_time
            
            # 记录指标
            current_step = epoch * num_batches + batch_idx
            self.metrics_tracker.add_metrics(
                current_step, 
                step_stats['avg_tokens'], 
                step_stats['avg_reward'], 
                step_stats['avg_loss']
            )
            
            epoch_metrics['token_counts'].append(step_stats['avg_tokens'])
            epoch_metrics['rewards'].append(step_stats['avg_reward'])
            epoch_metrics['losses'].append(step_stats['avg_loss'])
            
            # 日志输出
            if (batch_idx + 1) % self.config.logging_steps == 0:
                print(f"Epoch {epoch + 1}/{self.config.num_train_epochs}, "
                      f"Batch {batch_idx + 1}/{num_batches}, "
                      f"Avg Tokens: {step_stats['avg_tokens']:.1f}, "
                      f"Avg Reward: {step_stats['avg_reward']:.3f}, "
                      f"Loss: {step_stats['avg_loss']:.4f}, "
                      f"Cost Time: {cost_time:.2f}s")
                
                # 显示一个样例
                if len(batch) > 0:
                    sample_prompt = format_math_chat_input(batch[0]['question'], self.tokenizer)
                    sample_result = self.generate_response(sample_prompt)
                    print(f"样例问题: {batch[0]['question']}")
                    print(f"模型回答: {sample_result['response']}")
                    print(f"正确答案: {extract_answer(batch[0]['answer'])}")
                    print("-" * 50)
        
        # 返回epoch统计
        return {
            'avg_token_count': np.mean(epoch_metrics['token_counts']),
            'avg_reward': np.mean(epoch_metrics['rewards']),
            'avg_loss': np.mean(epoch_metrics['losses'])
        }

    def test_model(self):
        # 测试对话模式 - 使用数学问题
        print("\n" + "="*50)
        print("测试模型对话能力")
        print("="*50)
        
        # 构造对话格式的数学问题
        test_question = "小明有25个苹果，他给了朋友8个苹果，然后又买了12个苹果。请问小明现在有多少个苹果？"
        
        # 使用Qwen的对话格式
        messages = [
            {"role": "system", "content": "你是一个专业的数学助手，擅长解决各种数学问题。请逐步思考并给出准确答案。"},
            {"role": "user", "content": test_question}
        ]
        
        # 尝试使用chat template
        chat_input = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        print(f"输入prompt:\n{chat_input}\n")
        
        # tokenize对话输入
        inputs = self.tokenizer(chat_input, return_tensors="pt", truncation=True, max_length=self.config.max_length).to(self.device)
        
        with torch.no_grad():
            # 生成对话响应
            outputs = self.ref_model.generate(
                **inputs,
                max_new_tokens=self.config.max_length,  # 减少生成长度
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码并提取响应
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        print(f"模型回答:\n{response}\n")
        print("="*50 + "\n")
        
    def train(self):
        """主训练循环"""
        print("="*60)
        print("开始简化GRPO训练")
        print("="*60)
        
        # 设置模型和数据
        self.setup_models()
        self.setup_dataset()

        # self.test_model()

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
                os.makedirs(checkpoint_dir, exist_ok=True)
                self.model.save_pretrained(checkpoint_dir)
                self.tokenizer.save_pretrained(checkpoint_dir)
                print(f"检查点已保存到: {checkpoint_dir}")
        
        # 保存最终模型
        final_model_dir = os.path.join(self.config.output_dir, "final_model")
        self.model.save_pretrained(final_model_dir)
        self.tokenizer.save_pretrained(final_model_dir)
        print(f"最终模型已保存到: {final_model_dir}")
        
        # 保存训练指标
        metrics_dict = self.metrics_tracker.save_to_dict()
        save_metrics(metrics_dict, self.config.output_dir)
        
        print("训练完成！")

def main():
    """主函数"""
    print("初始化简化GRPO训练...")
    
    # 创建配置
    config = TrainingConfig()
    
    # 创建训练器
    trainer = SimpleGRPOTrainer(config)
    
    # 开始训练
    trainer.train()
    
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