#!/usr/bin/env python3
"""
GRPO训练快速启动脚本
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='GRPO训练快速启动')
    parser.add_argument('--action', choices=['train', 'visualize', 'check'], 
                      default='train', help='执行动作: train(训练), visualize(可视化), check(检查环境)')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--samples', type=int, default=1000, help='训练样本数')
    parser.add_argument('--batch-size', type=int, default=2, help='批处理大小')
    
    args = parser.parse_args()
    
    if args.action == 'check':
        check_environment()
    elif args.action == 'train':
        run_training(args.epochs, args.samples, args.batch_size)
    elif args.action == 'visualize':
        run_visualization()

def check_environment():
    """检查环境配置"""
    print("检查环境配置...")
    print("-" * 50)
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查依赖包
    required_packages = [
        'torch', 'transformers', 'trl', 'datasets', 
        'matplotlib', 'numpy', 'pandas'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: 已安装")
        except ImportError:
            print(f"❌ {package}: 未安装")
            missing_packages.append(package)
    
    # 检查PyTorch后端
    try:
        import torch
        print(f"\nPyTorch版本: {torch.__version__}")
        
        if torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon GPU): 可用")
        elif torch.cuda.is_available():
            print("✅ CUDA: 可用")
        else:
            print("⚠️  只有CPU可用")
            
    except ImportError:
        missing_packages.append('torch')
    
    if missing_packages:
        print(f"\n需要安装的包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ 环境检查通过！")
        return True

def run_training(epochs, samples, batch_size):
    """运行训练"""
    print(f"开始GRPO训练...")
    print(f"参数: epochs={epochs}, samples={samples}, batch_size={batch_size}")
    print("-" * 50)
    
    # 动态修改配置
    try:
        from config import TrainingConfig
        config = TrainingConfig()
        config.num_train_epochs = epochs
        config.max_samples = samples
        config.per_device_train_batch_size = batch_size
        
        # 导入并运行训练
        from train import GRPOTrainer
        trainer = GRPOTrainer(config)
        metrics = trainer.train()
        
        print("\n🎉 训练完成！")
        print("运行 'python run.py --action visualize' 查看结果")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        sys.exit(1)

def run_visualization():
    """运行可视化"""
    print("生成训练可视化...")
    print("-" * 50)
    
    try:
        from visualize import main as viz_main
        viz_main()
        print("\n🎉 可视化完成！")
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        print("请确保已完成训练并生成了指标文件")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 Qwen3 GRPO数学优化项目")
    print("=" * 50)
    main() 