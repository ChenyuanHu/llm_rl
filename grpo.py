import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from typing import Dict, List, Tuple
import numpy as np


class GRPOTrainer:
    """GRPO (Group Relative Policy Optimization) 训练器"""
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer,
        learning_rate: float = 1e-5,
        clip_ratio: float = 0.2,
        value_clip: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        gamma: float = 0.99,
        group_size: int = 4
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.clip_ratio = clip_ratio
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.group_size = group_size
        
        # 优化器
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # 添加价值头
        self.value_head = nn.Linear(model.config.hidden_size, 1).to(model.device)
        self.value_optimizer = torch.optim.AdamW(self.value_head.parameters(), lr=learning_rate)
        
        # 简化版本：不使用参考模型，直接使用当前模型的初始状态
        self.use_ref_model = False
        
    def generate_responses(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = False  # 使用贪心解码
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成响应"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False
            )
        
        return outputs.sequences, None
    
    def compute_log_probs(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        model = None
    ) -> torch.Tensor:
        """计算序列的对数概率"""
        if model is None:
            model = self.model
            
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # 计算每个token的对数概率
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 获取实际token的对数概率
        token_log_probs = torch.gather(
            log_probs[:, :-1], 
            dim=-1, 
            index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # 应用attention mask
        mask = attention_mask[:, 1:]
        token_log_probs = token_log_probs * mask
        
        return token_log_probs.sum(dim=-1)
    
    def compute_values(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """计算状态值"""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # 最后一层
            
            # 使用序列的最后一个非padding token的隐藏状态
            batch_size = hidden_states.size(0)
            seq_lengths = (attention_mask.sum(dim=-1) - 1).long()  # 确保是long类型
            last_hidden = hidden_states[range(batch_size), seq_lengths]
            
            values = self.value_head(last_hidden).squeeze(-1)
        
        return values
    
    def compute_group_relative_advantages(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor
    ) -> torch.Tensor:
        """计算组相对优势"""
        batch_size = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        
        # 按组计算相对优势
        for i in range(0, batch_size, self.group_size):
            end_idx = min(i + self.group_size, batch_size)
            group_rewards = rewards[i:end_idx]
            group_values = values[i:end_idx]
            
            # 计算组内平均奖励
            group_mean_reward = group_rewards.mean()
            
            # 相对优势 = 奖励 - 组平均奖励 - 值函数
            group_advantages = group_rewards - group_mean_reward - group_values
            advantages[i:end_idx] = group_advantages
        
        return advantages
    
    def compute_grpo_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """计算GRPO损失"""
        
        # 策略损失
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失
        value_pred = values
        value_target = rewards  # 简化版本，实际应该使用TD target
        value_loss_unclipped = F.mse_loss(value_pred, value_target, reduction='none')
        
        if self.value_clip > 0:
            value_pred_clipped = value_pred + torch.clamp(
                value_pred - value_pred,
                -self.value_clip,
                self.value_clip
            )
            value_loss_clipped = F.mse_loss(value_pred_clipped, value_target, reduction='none')
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
        else:
            value_loss = value_loss_unclipped.mean()
        
        # 熵损失（促进探索）
        entropy_loss = 0.0  # 简化版本，实际应该计算策略熵
        
        # 总损失
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss
        }
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
        rewards: torch.Tensor
    ) -> Dict[str, float]:
        """执行一步训练"""
        
        self.model.train()
        
        # 简化版本：使用当前模型计算旧概率（梯度停止）
        with torch.no_grad():
            old_log_probs = self.compute_log_probs(response_ids, response_mask, self.model)
        
        # 计算当前模型的对数概率
        new_log_probs = self.compute_log_probs(response_ids, response_mask, self.model)
        
        # 计算值函数
        values = self.compute_values(response_ids, response_mask)
        
        # 计算组相对优势
        advantages = self.compute_group_relative_advantages(rewards, values)
        
        # 计算损失
        losses = self.compute_grpo_loss(old_log_probs, new_log_probs, advantages, values, rewards)
        
        # 反向传播
        self.optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        
        losses["total_loss"].backward()
        
        # 梯度剪切
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        self.value_optimizer.step()
        
        # 返回损失信息
        return {
            "total_loss": losses["total_loss"].item(),
            "policy_loss": losses["policy_loss"].item(),
            "value_loss": losses["value_loss"].item(),
            "mean_reward": rewards.mean().item(),
            "mean_advantage": advantages.mean().item()
        } 