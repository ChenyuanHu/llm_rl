# Qwen3 0.5B GRPO数学优化项目

基于GRPO (Group Relative Policy Optimization) 算法优化Qwen3 0.5B模型在数学任务上的表现。

## 项目概述

本项目使用GRPO算法对Qwen3 0.5B模型进行强化学习训练，目标是提升模型在数学问题求解上的能力。训练过程包括：

- 使用GSM8K数学数据集
- 基于答案正确性的奖励机制
- 全面的训练指标监控和可视化

## 特性

✅ **Apple Silicon优化**: 针对M4 Pro芯片优化，支持MPS加速  
✅ **GRPO算法**: 使用TRL库实现的先进强化学习算法  
✅ **数学专用**: 专门针对数学推理任务优化  
✅ **完整监控**: 追踪Token使用、奖励、损失等关键指标  
✅ **可视化分析**: 自动生成训练过程图表和分析报告  

## 环境要求

- Python 3.8+
- PyTorch 2.1.0+
- Apple Silicon Mac (推荐) 或其他GPU设备
- 48GB+ RAM (推荐)

## 安装依赖

1. 激活虚拟环境:
```bash
source venv/bin/activate
```

2. 安装依赖包:
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 开始训练

```bash
python train.py
```

训练配置可在 `config.py` 中修改：
- `num_train_epochs`: 训练轮数 (默认: 3)
- `max_samples`: 训练样本数量 (默认: 1000)
- `per_device_train_batch_size`: 批处理大小 (默认: 2)
- `learning_rate`: 学习率 (默认: 5e-6)

### 2. 查看训练结果

训练完成后，运行可视化脚本：

```bash
python visualize.py
```

这将生成：
- 训练趋势图 (`outputs/training_metrics.png`)
- 指标分布图 (`outputs/metrics_distribution.png`)
- 训练报告 (`outputs/training_report.json`)

## 项目结构

```
qwen_rl/
├── config.py          # 训练配置
├── train.py           # 主训练脚本
├── utils.py           # 工具函数
├── visualize.py       # 可视化脚本
├── requirements.txt   # 依赖包
├── README.md         # 项目说明
├── outputs/          # 训练输出
│   ├── final_model/     # 最终模型
│   ├── checkpoint-*/    # 训练检查点
│   ├── training_metrics.json # 训练指标
│   └── *.png           # 可视化图表
└── logs/             # 训练日志
```

## 核心功能

### GRPO训练器
- 基于PPO算法的GRPO实现
- 自动化的奖励计算和指标追踪
- Apple Silicon优化的模型加载和训练

### 数据处理
- GSM8K数学数据集自动下载和处理
- 智能答案提取和验证
- 中文数学问题格式化

### 可视化分析
- 实时训练指标监控
- Token使用量趋势分析
- 奖励和损失变化可视化
- 详细的统计报告生成

## 监控指标

训练过程中追踪以下关键指标：

1. **Token数量**: 每个问题解答的平均Token使用量
2. **奖励分数**: 基于答案正确性的奖励值
3. **训练损失**: GRPO算法的训练损失
4. **模型性能**: 答案准确率和响应质量

## 配置说明

### 主要训练参数

- `model_name`: 基础模型 (Qwen/Qwen2.5-0.5B)
- `dataset_name`: 数据集 (openai/gsm8k)
- `learning_rate`: 学习率 (5e-6)
- `beta`: KL散度惩罚系数 (0.1)
- `max_length`: 最大序列长度 (512)

### Apple Silicon优化

- 自动检测MPS可用性
- 优化的批处理大小和工作进程数
- 内存高效的模型加载策略

## 故障排除

### 常见问题

1. **内存不足**: 减少 `per_device_train_batch_size` 或 `max_samples`
2. **MPS不可用**: 训练将自动降级到CPU模式
3. **依赖冲突**: 使用虚拟环境并按要求安装依赖

### 性能优化

- 确保使用MPS加速 (Apple Silicon)
- 根据可用内存调整批处理大小
- 使用合适的数据类型 (float16)

## 训练结果示例

训练完成后，您将看到类似以下的结果：

```
训练总结:
平均Token数: 45.2
平均Reward: 0.342
平均Loss: 0.0156

运行以下命令查看训练可视化:
python visualize.py
```

## 扩展功能

项目支持以下扩展：

- 自定义数学数据集
- 不同的奖励函数设计
- 多种评估指标
- WandB日志集成
- 分布式训练支持

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

MIT License 