#!/usr/bin/env python3
"""
简化的GRPO训练启动脚本
"""

import sys
import os

def main():
    print("=" * 60)
    print("🚀 GRPO Qwen3数学推理优化项目")
    print("=" * 60)
    print()
    
    print("📋 项目配置:")
    print("  - 模型: Qwen2.5-0.5B")
    print("  - 数据集: GSM8K")
    print("  - 算法: GRPO")
    print("  - 训练轮数: 3")
    print("  - 批次大小: 4")
    print()
    
    # 检查依赖
    try:
        import torch
        import transformers
        import datasets
        print("✅ 依赖检查通过")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        sys.exit(1)
    
    # 检查设备
    if torch.backends.mps.is_available():
        print("🖥️  检测到Apple Silicon MPS")
    elif torch.cuda.is_available():
        print(f"🖥️  检测到CUDA GPU: {torch.cuda.get_device_name()}")
    else:
        print("🖥️  使用CPU")
    
    print()
    print("🔄 开始训练...")
    print("=" * 60)
    
    # 运行训练
    from train import main as train_main
    train_main()

if __name__ == "__main__":
    main() 