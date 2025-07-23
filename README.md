# GRPO优化Qwen3数学推理项目

本项目使用GRPO (Group Relative Policy Optimization) 算法来优化Qwen3 0.5B模型在数学问题解答方面的表现。

## 项目结构

```
qwen_rl/
├── requirements.txt      # 项目依赖
├── utils.py             # 工具函数（数据处理、奖励计算等）
├── grpo.py              # GRPO算法实现
├── train.py             # 主训练脚本
├── visualize.py         # 可视化工具
├── run.py               # 简化启动脚本
└── outputs/             # 输出目录（模型、指标、图表）
```

## 功能特性

- 🤖 **模型**: Qwen2.5-0.5B (约0.6B参数)
- 📊 **数据集**: GSM8K数学应用题数据集
- 🎯 **算法**: GRPO强化学习算法
- 📈 **可视化**: 训练损失、奖励和token数量趋势图
- 💾 **自动保存**: 训练指标、模型检查点

## 训练配置

- **训练轮数**: 3 epochs
- **批次大小**: 4
- **学习率**: 1e-5
- **最大新token数**: 256
- **组大小**: 4 (用于GRPO)
- **奖励机制**: 基于最终答案正确性的简单奖励

## 使用说明

1. **环境准备**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **开始训练**:
   ```bash
   python train.py
   ```

3. **查看结果**:
   训练完成后，在 `outputs/` 目录中可以找到：
   - `training_metrics.png`: 训练指标趋势图
   - `metrics.json`: 详细训练数据
   - `final_model/`: 训练好的模型
   - `training_report.txt`: 训练总结报告

## 输出指标

训练过程中会跟踪并可视化以下指标：

1. **训练损失**: 总体训练损失的变化趋势
2. **平均奖励**: 模型回答正确率的提升
3. **平均Token数**: 每个回答的token数量变化
4. **综合比较**: 所有指标的归一化比较图

## 硬件要求

- **内存**: 至少16GB RAM (推荐32GB+)
- **处理器**: 支持MPS的Apple Silicon或CUDA GPU
- **存储**: 至少10GB可用空间

## 算法说明

GRPO (Group Relative Policy Optimization) 是一种改进的强化学习算法，通过以下方式优化模型：

1. **组相对比较**: 在小组内比较不同响应的质量
2. **策略梯度**: 使用PPO风格的剪切更新
3. **价值函数**: 估计状态值以减少方差
4. **奖励计算**: 基于数学问题答案的正确性

## 自定义配置

可以在 `train.py` 中修改 `config` 字典来调整训练参数：

```python
config = {
    "model_name": "Qwen/Qwen2.5-0.5B",
    "max_samples_train": 1000,  # 增加训练样本
    "batch_size": 8,            # 调整批次大小
    "num_epochs": 5,            # 增加训练轮数
    "learning_rate": 2e-5,      # 调整学习率
    # ...
}
```

## 故障排除

1. **内存不足**: 减少 `batch_size` 或 `max_samples_train`
2. **CUDA错误**: 确保PyTorch版本与CUDA版本匹配
3. **下载失败**: 检查网络连接，可能需要设置HuggingFace镜像

## 许可证

MIT License 