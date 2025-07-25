# 策略损失函数详解

## 🎯 1. 策略梯度的理论基础

### 强化学习目标
在语言模型的强化学习中，我们的目标是：
**最大化生成文本的期望奖励**

```
J(θ) = E_{τ~π_θ}[R(τ)]
```

其中：
- `θ`: 模型参数
- `π_θ`: 参数为θ的策略（即我们的语言模型）
- `τ`: 轨迹（在NLP中就是生成的文本序列）
- `R(τ)`: 奖励（例如：答案是否正确）

### 策略梯度公式
通过数学推导，策略梯度为：

```
∇_θ J(θ) = E_{τ~π_θ}[∇_θ log π_θ(τ) × R(τ)]
```

对于序列生成，这等价于：

```
∇_θ J(θ) = E[∑_{t=1}^T ∇_θ log π_θ(a_t|s_t) × R]
```

## 🔍 2. compute_policy_loss 函数逐行解析

### 第一步：准备数据
```python
def compute_policy_loss(self, input_ids, response_ids, rewards):
    # 确保response_ids有正确的形状
    if response_ids.dim() == 1:
        response_ids = response_ids.unsqueeze(0)  # 添加batch维度
    
    # 准备输入：拼接prompt和response
    full_ids = torch.cat([input_ids, response_ids], dim=1)
```

**作用**：
- 将prompt（问题）和response（模型生成的答案）拼接成完整序列
- 确保tensor维度正确（添加batch维度）

**示例**：
```
input_ids:    [1, 2, 3, 4]     # 问题："计算1+1"的token ids
response_ids: [5, 6]           # 答案："等于2"的token ids  
full_ids:     [1, 2, 3, 4, 5, 6]  # 完整对话
```

### 第二步：前向传播
```python
# 前向传播 - 不使用labels来避免自动计算loss
outputs = self.model(full_ids)
logits = outputs.logits
```

**作用**：
- 让模型处理完整序列
- 获取每个位置的logits（词汇表上的概率分布）

**logits含义**：
```
logits.shape: [batch_size, sequence_length, vocab_size]
logits[0, i, :] 表示位置i预测下一个token的概率分布
```

### 第三步：提取响应部分的logits
```python
prompt_length = input_ids.shape[1]
response_length = response_ids.shape[1]

# 获取response部分的logits（用于预测下一个token）
response_logits = logits[:, prompt_length-1:prompt_length+response_length-1]
```

**关键理解**：
在语言模型中，`logits[i]` 用于预测 `token[i+1]`

```
序列:      [问题tokens...] [答案token1] [答案token2] [答案token3]
位置:           0  1  2         3         4         5
logits用途:   预测1 预测2 预测3    预测4     预测5     预测6

要计算答案tokens的损失，需要：
- logits[2] 预测 答案token1
- logits[3] 预测 答案token2  
- logits[4] 预测 答案token3
```

### 第四步：计算log概率
```python
# 计算log概率
log_probs = F.log_softmax(response_logits, dim=-1)

# 选择实际生成的token的log概率
selected_log_probs = log_probs[0].gather(1, response_ids[0].unsqueeze(1)).squeeze()
current_log_prob = selected_log_probs.mean()
```

**作用**：
1. `log_softmax`: 将logits转换为log概率
2. `gather`: 提取模型对实际生成token的log概率
3. `mean`: 计算平均log概率

**数学含义**：
```
log π_θ(response) = ∑_t log π_θ(token_t | context_t)
```

### 第五步：计算策略损失
```python
# 策略梯度损失：最大化奖励加权的log概率
policy_loss = -current_log_prob * rewards
```

**核心公式**：
```
Loss = -log π_θ(action) × reward
```

**原理解释**：
- 如果 `reward > 0`（答案正确）：loss为负，梯度上升，增加生成该序列的概率
- 如果 `reward < 0`（答案错误）：loss为正，梯度下降，减少生成该序列的概率

## 🧮 3. 数学原理详解

### 为什么是负号？
在PyTorch中，优化器默认最小化损失函数。但我们想要**最大化**期望奖励：

```
目标：max E[log π(a) × R]
等价于：min E[-log π(a) × R]  # 添加负号转换为最小化问题
```

### 为什么用log概率？
1. **数值稳定性**：避免概率连乘导致的数值下溢
2. **计算效率**：log将乘法转换为加法
3. **梯度形式**：∇log π(a) 有更好的梯度特性

### GRPO vs 传统策略梯度
传统策略梯度可能有高方差问题，GRPO/PPO通过以下方式改进：

1. **重要性采样**：使用旧策略和新策略的比值
2. **clip机制**：限制策略更新幅度
3. **baseline减法**：减少方差

我们这里实现的是简化版本，主要用于理解核心概念。

## 🎯 4. 实际效果

当模型训练时：

**正确答案场景**：
```
问题: "1+1等于多少？"
模型回答: "等于2"  
奖励: +1.0
结果: 增加生成"等于2"的概率
```

**错误答案场景**：
```
问题: "1+1等于多少？"
模型回答: "等于3"
奖励: -0.1  
结果: 减少生成"等于3"的概率
```

通过不断这样的训练，模型学会生成更高奖励的回答！

## 🔧 5. 代码改进建议

当前实现可以进一步优化：

1. **添加entropy正则化**：鼓励探索
2. **使用advantage函数**：减少方差
3. **实现PPO的clip机制**：稳定训练
4. **批处理优化**：提高效率

这就是策略损失的核心原理！🚀 