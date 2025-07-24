import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List
import os

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_training_metrics(metrics_file: str = "./outputs/training_metrics.json") -> Dict:
    """加载训练指标数据"""
    if not os.path.exists(metrics_file):
        print(f"指标文件 {metrics_file} 不存在")
        return None
    
    with open(metrics_file, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    print(f"成功加载训练指标，共 {len(metrics['steps'])} 个数据点")
    return metrics

def smooth_curve(data: List[float], window_size: int = 5) -> List[float]:
    """平滑曲线"""
    if len(data) < window_size:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    
    return smoothed

def plot_training_metrics(metrics: Dict, save_dir: str = "./outputs"):
    """绘制训练指标图表"""
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取数据
    steps = metrics['steps']
    token_counts = metrics['token_counts']
    rewards = metrics['rewards']
    losses = metrics['losses']
    
    # 平滑数据
    smoothed_tokens = smooth_curve(token_counts)
    smoothed_rewards = smooth_curve(rewards)
    smoothed_losses = smooth_curve(losses)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('GRPO训练过程指标监控', fontsize=16, fontweight='bold')
    
    # 1. 平均Token数量趋势
    ax1 = axes[0, 0]
    ax1.plot(steps, token_counts, alpha=0.3, color='blue', label='原始数据')
    ax1.plot(steps, smoothed_tokens, color='blue', linewidth=2, label='平滑趋势')
    ax1.set_title('平均每个问题的解答Token数量趋势', fontsize=12, fontweight='bold')
    ax1.set_xlabel('训练步数')
    ax1.set_ylabel('Token数量')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_tokens = np.mean(token_counts)
    ax1.axhline(y=mean_tokens, color='red', linestyle='--', alpha=0.7, label=f'平均值: {mean_tokens:.1f}')
    ax1.legend()
    
    # 2. 奖励趋势
    ax2 = axes[0, 1]
    ax2.plot(steps, rewards, alpha=0.3, color='green', label='原始数据')
    ax2.plot(steps, smoothed_rewards, color='green', linewidth=2, label='平滑趋势')
    ax2.set_title('奖励(Reward)趋势', fontsize=12, fontweight='bold')
    ax2.set_xlabel('训练步数')
    ax2.set_ylabel('平均奖励')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_reward = np.mean(rewards)
    ax2.axhline(y=mean_reward, color='red', linestyle='--', alpha=0.7, label=f'平均值: {mean_reward:.3f}')
    ax2.legend()
    
    # 3. Loss趋势
    ax3 = axes[1, 0]
    ax3.plot(steps, losses, alpha=0.3, color='orange', label='原始数据')
    ax3.plot(steps, smoothed_losses, color='orange', linewidth=2, label='平滑趋势')
    ax3.set_title('损失(Loss)趋势', fontsize=12, fontweight='bold')
    ax3.set_xlabel('训练步数')
    ax3.set_ylabel('平均损失')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 综合指标对比
    ax4 = axes[1, 1]
    
    # 归一化数据进行对比
    norm_tokens = np.array(smoothed_tokens) / np.max(smoothed_tokens)
    norm_rewards = (np.array(smoothed_rewards) - np.min(smoothed_rewards)) / (np.max(smoothed_rewards) - np.min(smoothed_rewards))
    norm_losses = 1 - (np.array(smoothed_losses) - np.min(smoothed_losses)) / (np.max(smoothed_losses) - np.min(smoothed_losses))  # 反转loss
    
    ax4.plot(steps, norm_tokens, label='Token数量(归一化)', linewidth=2)
    ax4.plot(steps, norm_rewards, label='奖励(归一化)', linewidth=2)
    ax4.plot(steps, norm_losses, label='Loss(反转归一化)', linewidth=2)
    ax4.set_title('综合指标对比(归一化)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('训练步数')
    ax4.set_ylabel('归一化值')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(save_dir, "training_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"训练指标图表已保存到: {plot_path}")
    
    # 显示图表
    plt.show()

def plot_distribution_analysis(metrics: Dict, save_dir: str = "./outputs"):
    """绘制指标分布分析图"""
    
    token_counts = metrics['token_counts']
    rewards = metrics['rewards']
    losses = metrics['losses']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('训练指标分布分析', fontsize=16, fontweight='bold')
    
    # Token数量分布
    ax1 = axes[0]
    ax1.hist(token_counts, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.mean(token_counts), color='red', linestyle='--', linewidth=2, label=f'均值: {np.mean(token_counts):.1f}')
    ax1.axvline(np.median(token_counts), color='green', linestyle='--', linewidth=2, label=f'中位数: {np.median(token_counts):.1f}')
    ax1.set_title('Token数量分布')
    ax1.set_xlabel('Token数量')
    ax1.set_ylabel('频次')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 奖励分布
    ax2 = axes[1]
    ax2.hist(rewards, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'均值: {np.mean(rewards):.3f}')
    ax2.axvline(np.median(rewards), color='blue', linestyle='--', linewidth=2, label=f'中位数: {np.median(rewards):.3f}')
    ax2.set_title('奖励分布')
    ax2.set_xlabel('奖励值')
    ax2.set_ylabel('频次')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Loss分布
    ax3 = axes[2]
    ax3.hist(losses, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(np.mean(losses), color='red', linestyle='--', linewidth=2, label=f'均值: {np.mean(losses):.4f}')
    ax3.axvline(np.median(losses), color='blue', linestyle='--', linewidth=2, label=f'中位数: {np.median(losses):.4f}')
    ax3.set_title('Loss分布')
    ax3.set_xlabel('Loss值')
    ax3.set_ylabel('频次')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    dist_path = os.path.join(save_dir, "metrics_distribution.png")
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    print(f"指标分布图已保存到: {dist_path}")
    
    plt.show()

def generate_training_report(metrics: Dict, save_dir: str = "./outputs"):
    """生成训练报告"""
    
    token_counts = metrics['token_counts']
    rewards = metrics['rewards']
    losses = metrics['losses']
    steps = metrics['steps']
    
    report = {
        "训练概况": {
            "总训练步数": len(steps),
            "训练步数范围": f"{min(steps)} - {max(steps)}"
        },
        "Token使用统计": {
            "平均Token数": round(np.mean(token_counts), 2),
            "Token数中位数": round(np.median(token_counts), 2),
            "Token数标准差": round(np.std(token_counts), 2),
            "最小Token数": min(token_counts),
            "最大Token数": max(token_counts)
        },
        "奖励统计": {
            "平均奖励": round(np.mean(rewards), 4),
            "奖励中位数": round(np.median(rewards), 4),
            "奖励标准差": round(np.std(rewards), 4),
            "最小奖励": round(min(rewards), 4),
            "最大奖励": round(max(rewards), 4),
            "正奖励比例": f"{sum(1 for r in rewards if r > 0) / len(rewards) * 100:.1f}%"
        },
        "Loss统计": {
            "平均Loss": round(np.mean(losses), 6),
            "Loss中位数": round(np.median(losses), 6),
            "Loss标准差": round(np.std(losses), 6),
            "最小Loss": round(min(losses), 6),
            "最大Loss": round(max(losses), 6)
        }
    }
    
    # 保存报告
    report_path = os.path.join(save_dir, "training_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印报告
    print("\n" + "="*60)
    print("训练报告")
    print("="*60)
    
    for category, stats in report.items():
        print(f"\n{category}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    print(f"\n详细报告已保存到: {report_path}")

def main():
    """主函数"""
    print("开始生成训练可视化...")
    
    # 加载指标数据
    metrics = load_training_metrics()
    
    if metrics is None:
        print("无法加载训练指标，请确保已完成训练")
        return
    
    # 生成可视化图表
    print("\n生成训练趋势图...")
    plot_training_metrics(metrics)
    
    print("\n生成指标分布图...")
    plot_distribution_analysis(metrics)
    
    print("\n生成训练报告...")
    generate_training_report(metrics)
    
    print("\n可视化完成！")
    print("生成的文件:")
    print("- ./outputs/training_metrics.png")
    print("- ./outputs/metrics_distribution.png") 
    print("- ./outputs/training_report.json")

if __name__ == "__main__":
    main() 