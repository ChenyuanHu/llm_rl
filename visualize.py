import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_training_metrics(metrics: Dict, save_dir: str):
    """绘制训练指标趋势图"""
    
    epochs = metrics.get("epochs", [])
    losses = metrics.get("losses", [])
    rewards = metrics.get("rewards", [])
    token_counts = metrics.get("token_counts", [])
    
    if not epochs:
        print("没有训练指标数据可供绘制")
        return
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 损失趋势
    ax1.plot(epochs, losses, 'b-', linewidth=2, marker='o')
    ax1.set_title('训练损失趋势', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # 2. 奖励趋势
    ax2.plot(epochs, rewards, 'g-', linewidth=2, marker='s')
    ax2.set_title('平均奖励趋势', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Average Reward')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # 3. Token数量趋势
    ax3.plot(epochs, token_counts, 'r-', linewidth=2, marker='^')
    ax3.set_title('平均Token数量趋势', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Average Token Count')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)
    
    # 4. 综合指标
    if len(epochs) > 1:
        # 归一化指标用于比较
        norm_losses = np.array(losses) / max(losses) if max(losses) > 0 else np.array(losses)
        norm_rewards = np.array(rewards)
        norm_tokens = np.array(token_counts) / max(token_counts) if max(token_counts) > 0 else np.array(token_counts)
        
        ax4.plot(epochs, norm_losses, 'b-', label='损失 (归一化)', linewidth=2)
        ax4.plot(epochs, norm_rewards, 'g-', label='奖励', linewidth=2)
        ax4.plot(epochs, norm_tokens, 'r-', label='Token数 (归一化)', linewidth=2)
        ax4.set_title('综合指标比较', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Normalized Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.1)
    else:
        ax4.text(0.5, 0.5, '需要更多epoch数据\n才能显示比较图', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('综合指标比较', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, "training_metrics.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"训练指标图已保存到: {save_path}")
    
    plt.close()


def plot_detailed_metrics(metrics: Dict, save_dir: str):
    """绘制详细的训练指标"""
    
    epochs = metrics.get("epochs", [])
    losses = metrics.get("losses", [])
    rewards = metrics.get("rewards", [])
    token_counts = metrics.get("token_counts", [])
    
    if not epochs:
        return
    
    # 损失详细分析
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=6)
    ax.set_title('训练损失详细趋势', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 添加趋势线
    if len(epochs) > 2:
        z = np.polyfit(epochs, losses, 1)
        p = np.poly1d(z)
        ax.plot(epochs, p(epochs), "r--", alpha=0.8, label=f'趋势线 (斜率: {z[0]:.4f})')
        ax.legend()
    
    # 添加数值标注
    for i, (x, y) in enumerate(zip(epochs, losses)):
        if i % max(1, len(epochs)//5) == 0:  # 每隔几个点标注一次
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_detailed.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 奖励详细分析
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, rewards, 'g-', linewidth=2, marker='s', markersize=6)
    ax.set_title('奖励详细趋势', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # 添加改进幅度
    if len(rewards) > 1:
        improvement = rewards[-1] - rewards[0]
        ax.text(0.02, 0.98, f'总改进: {improvement:.3f}', 
                transform=ax.transAxes, fontsize=12, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "reward_detailed.png"), dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_report(metrics: Dict, config: Dict, save_dir: str):
    """创建训练总结报告"""
    
    epochs = metrics.get("epochs", [])
    losses = metrics.get("losses", [])
    rewards = metrics.get("rewards", [])
    token_counts = metrics.get("token_counts", [])
    
    if not epochs:
        return
    
    # 生成文本报告
    report = []
    report.append("=" * 50)
    report.append("GRPO训练总结报告")
    report.append("=" * 50)
    report.append("")
    
    # 配置信息
    report.append("训练配置:")
    report.append(f"  模型: {config.get('model_name', 'N/A')}")
    report.append(f"  Epoch数: {config.get('num_epochs', 'N/A')}")
    report.append(f"  批次大小: {config.get('batch_size', 'N/A')}")
    report.append(f"  学习率: {config.get('learning_rate', 'N/A')}")
    report.append(f"  组大小: {config.get('group_size', 'N/A')}")
    report.append("")
    
    # 训练结果
    report.append("训练结果:")
    if losses:
        report.append(f"  初始损失: {losses[0]:.4f}")
        report.append(f"  最终损失: {losses[-1]:.4f}")
        report.append(f"  损失改进: {losses[0] - losses[-1]:.4f}")
    
    if rewards:
        report.append(f"  初始奖励: {rewards[0]:.4f}")
        report.append(f"  最终奖励: {rewards[-1]:.4f}")
        report.append(f"  奖励改进: {rewards[-1] - rewards[0]:.4f}")
    
    if token_counts:
        report.append(f"  平均Token数: {np.mean(token_counts):.2f}")
        report.append(f"  Token数标准差: {np.std(token_counts):.2f}")
    
    report.append("")
    report.append("训练完成时间: " + str(np.datetime64('now')))
    
    # 保存报告
    report_path = os.path.join(save_dir, "training_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"训练报告已保存到: {report_path}")


def plot_token_distribution(token_counts: List[float], save_dir: str):
    """绘制token数量分布图"""
    
    if not token_counts:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 直方图
    ax1.hist(token_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Token数量分布', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Token Count')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # 箱线图
    ax2.boxplot(token_counts, patch_artist=True,
                boxprops=dict(facecolor='lightgreen', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_title('Token数量箱线图', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Token Count')
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f'均值: {np.mean(token_counts):.1f}\n中位数: {np.median(token_counts):.1f}\n标准差: {np.std(token_counts):.1f}'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "token_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Token分布图已保存到: {os.path.join(save_dir, 'token_distribution.png')}") 