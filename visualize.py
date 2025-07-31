import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List
import os

# 设置英文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_training_metrics(metrics_file: str = "./outputs/training_metrics.json") -> Dict:
    """Load training metrics data"""
    if not os.path.exists(metrics_file):
        print(f"Metrics file {metrics_file} does not exist")
        return None
    
    with open(metrics_file, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    print(f"Successfully loaded training metrics with {len(metrics['steps'])} data points")
    return metrics

def smooth_curve(data: List[float], window_size: int = 5) -> List[float]:
    """Smooth curve using moving average"""
    if len(data) < window_size:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    
    return smoothed

def plot_training_metrics(metrics: Dict, save_dir: str = "./outputs"):
    """Plot training metrics charts"""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    steps = metrics['steps']
    token_counts = metrics['token_counts']
    rewards = metrics['rewards']
    losses = metrics['losses']
    
    # 检查是否有评估数据
    has_eval_data = 'eval_rewards' in metrics and len(metrics['eval_rewards']) > 0
    if has_eval_data:
        eval_rewards = metrics['eval_rewards']
        eval_token_counts = metrics['eval_token_counts']
        smoothed_eval_rewards = smooth_curve(eval_rewards)
        smoothed_eval_tokens = smooth_curve(eval_token_counts)
    
    # Smooth data
    smoothed_tokens = smooth_curve(token_counts)
    smoothed_rewards = smooth_curve(rewards)
    smoothed_losses = smooth_curve(losses)
    
    # Create charts - 如果有评估数据，创建2x3布局，否则保持2x2
    if has_eval_data:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('GRPO Training Process Metrics Monitoring', fontsize=16, fontweight='bold')
    
    # 1. Average Token Count Trend
    ax1 = axes[0, 0]
    ax1.plot(steps, token_counts, alpha=0.3, color='blue', label='Raw Data')
    ax1.plot(steps, smoothed_tokens, color='blue', linewidth=2, label='Smoothed Trend')
    ax1.set_title('Average Token Count per Response', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Token Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistical information
    mean_tokens = np.mean(token_counts)
    ax1.axhline(y=mean_tokens, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_tokens:.1f}')
    ax1.legend()
    
    # 2. Training vs Evaluation Reward Comparison
    ax2 = axes[0, 1]
    ax2.plot(steps, rewards, alpha=0.3, color='green', label='Train Reward (Raw)')
    ax2.plot(steps, smoothed_rewards, color='green', linewidth=2, label='Train Reward (Smoothed)')
    
    if has_eval_data:
        ax2.plot(steps, eval_rewards, alpha=0.3, color='purple', label='Eval Reward (Raw)')
        ax2.plot(steps, smoothed_eval_rewards, color='purple', linewidth=2, label='Eval Reward (Smoothed)')
    
    ax2.set_title('Training vs Evaluation Reward Trend', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Average Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistical information
    mean_reward = np.mean(rewards)
    ax2.axhline(y=mean_reward, color='red', linestyle='--', alpha=0.7, label=f'Train Mean: {mean_reward:.3f}')
    if has_eval_data:
        mean_eval_reward = np.mean(eval_rewards)
        ax2.axhline(y=mean_eval_reward, color='orange', linestyle='--', alpha=0.7, label=f'Eval Mean: {mean_eval_reward:.3f}')
    ax2.legend()
    
    # 3. Loss Trend
    ax3_pos = (1, 0) if has_eval_data else (1, 0)
    ax3 = axes[ax3_pos[0], ax3_pos[1]]
    ax3.plot(steps, losses, alpha=0.3, color='orange', label='Raw Data')
    ax3.plot(steps, smoothed_losses, color='orange', linewidth=2, label='Smoothed Trend')
    ax3.set_title('Loss Trend', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Average Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 如果有评估数据，添加评估奖励分布图
    if has_eval_data:
        ax4 = axes[0, 2]
        ax4.plot(steps, eval_rewards, alpha=0.3, color='purple', label='Raw Data')
        ax4.plot(steps, smoothed_eval_rewards, color='purple', linewidth=2, label='Smoothed Trend')
        ax4.set_title('Evaluation Reward Trend', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Evaluation Reward')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add statistical information
        mean_eval_reward = np.mean(eval_rewards)
        ax4.axhline(y=mean_eval_reward, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_eval_reward:.3f}')
        ax4.legend()
        
        # 5. Evaluation Token Count Trend
        ax5 = axes[1, 1]
        ax5.plot(steps, eval_token_counts, alpha=0.3, color='cyan', label='Raw Data')
        ax5.plot(steps, smoothed_eval_tokens, color='cyan', linewidth=2, label='Smoothed Trend')
        ax5.set_title('Evaluation Token Count Trend', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Training Steps')
        ax5.set_ylabel('Evaluation Token Count')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Comprehensive Metrics Comparison
        ax6 = axes[1, 2]
        
        # Normalize data for comparison
        norm_tokens = np.array(smoothed_tokens) / np.max(smoothed_tokens)
        norm_rewards = (np.array(smoothed_rewards) - np.min(smoothed_rewards)) / (np.max(smoothed_rewards) - np.min(smoothed_rewards))
        norm_eval_rewards = (np.array(smoothed_eval_rewards) - np.min(smoothed_eval_rewards)) / (np.max(smoothed_eval_rewards) - np.min(smoothed_eval_rewards))
        norm_losses = 1 - (np.array(smoothed_losses) - np.min(smoothed_losses)) / (np.max(smoothed_losses) - np.min(smoothed_losses))  # Invert loss
        
        ax6.plot(steps, norm_tokens, label='Token Count (Normalized)', linewidth=2)
        ax6.plot(steps, norm_rewards, label='Train Reward (Normalized)', linewidth=2)
        ax6.plot(steps, norm_eval_rewards, label='Eval Reward (Normalized)', linewidth=2)
        ax6.plot(steps, norm_losses, label='Loss (Inverted Normalized)', linewidth=2)
        ax6.set_title('Comprehensive Metrics Comparison (Normalized)', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Training Steps')
        ax6.set_ylabel('Normalized Value')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        # 4. Comprehensive Metrics Comparison (original layout)
        ax4 = axes[1, 1]
        
        # Normalize data for comparison
        norm_tokens = np.array(smoothed_tokens) / np.max(smoothed_tokens)
        norm_rewards = (np.array(smoothed_rewards) - np.min(smoothed_rewards)) / (np.max(smoothed_rewards) - np.min(smoothed_rewards))
        norm_losses = 1 - (np.array(smoothed_losses) - np.min(smoothed_losses)) / (np.max(smoothed_losses) - np.min(smoothed_losses))  # Invert loss
        
        ax4.plot(steps, norm_tokens, label='Token Count (Normalized)', linewidth=2)
        ax4.plot(steps, norm_rewards, label='Reward (Normalized)', linewidth=2)
        ax4.plot(steps, norm_losses, label='Loss (Inverted Normalized)', linewidth=2)
        ax4.set_title('Comprehensive Metrics Comparison (Normalized)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Normalized Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart
    plot_path = os.path.join(save_dir, "training_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training metrics chart saved to: {plot_path}")
    
    # Display chart
    plt.show()

def plot_distribution_analysis(metrics: Dict, save_dir: str = "./outputs"):
    """Plot metrics distribution analysis"""
    
    token_counts = metrics['token_counts']
    rewards = metrics['rewards']
    losses = metrics['losses']
    
    # 检查是否有评估数据
    has_eval_data = 'eval_rewards' in metrics and len(metrics['eval_rewards']) > 0
    if has_eval_data:
        eval_rewards = metrics['eval_rewards']
        eval_token_counts = metrics['eval_token_counts']
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    fig.suptitle('Training Metrics Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 调整坐标系索引
    if has_eval_data:
        # 如果有评估数据，使用2x3布局
        # 第一行：训练数据分布
        ax1 = axes[0, 0]
        ax2 = axes[0, 1]  
        ax3 = axes[0, 2]
        
        # 第二行：评估数据分布
        ax4 = axes[1, 0]
        ax5 = axes[1, 1]
        ax6 = axes[1, 2]
    else:
        # 如果没有评估数据，使用1x3布局
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]
    
    # Token Count Distribution
    ax1.hist(token_counts, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.mean(token_counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(token_counts):.1f}')
    ax1.axvline(np.median(token_counts), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(token_counts):.1f}')
    ax1.set_title('Training Token Count Distribution')
    ax1.set_xlabel('Token Count')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Reward Distribution
    ax2.hist(rewards, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.3f}')
    ax2.axvline(np.median(rewards), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.3f}')
    ax2.set_title('Training Reward Distribution')
    ax2.set_xlabel('Reward Value')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Loss Distribution
    ax3.hist(losses, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(np.mean(losses), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(losses):.4f}')
    ax3.axvline(np.median(losses), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(losses):.4f}')
    ax3.set_title('Loss Distribution')
    ax3.set_xlabel('Loss Value')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 如果有评估数据，添加评估分布图
    if has_eval_data:
        # Evaluation Token Count Distribution
        ax4.hist(eval_token_counts, bins=30, alpha=0.7, color='cyan', edgecolor='black')
        ax4.axvline(np.mean(eval_token_counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(eval_token_counts):.1f}')
        ax4.axvline(np.median(eval_token_counts), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(eval_token_counts):.1f}')
        ax4.set_title('Evaluation Token Count Distribution')
        ax4.set_xlabel('Token Count')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Evaluation Reward Distribution
        ax5.hist(eval_rewards, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax5.axvline(np.mean(eval_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(eval_rewards):.3f}')
        ax5.axvline(np.median(eval_rewards), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(eval_rewards):.3f}')
        ax5.set_title('Evaluation Reward Distribution')
        ax5.set_xlabel('Reward Value')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Training vs Evaluation Reward Comparison
        ax6.hist(rewards, bins=30, alpha=0.5, color='green', edgecolor='black', label='Training Rewards')
        ax6.hist(eval_rewards, bins=30, alpha=0.5, color='purple', edgecolor='black', label='Evaluation Rewards')
        ax6.axvline(np.mean(rewards), color='darkgreen', linestyle='--', linewidth=2, label=f'Train Mean: {np.mean(rewards):.3f}')
        ax6.axvline(np.mean(eval_rewards), color='darkmagenta', linestyle='--', linewidth=2, label=f'Eval Mean: {np.mean(eval_rewards):.3f}')
        ax6.set_title('Training vs Evaluation Reward Comparison')
        ax6.set_xlabel('Reward Value')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart
    dist_path = os.path.join(save_dir, "metrics_distribution.png")
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    print(f"Metrics distribution chart saved to: {dist_path}")
    
    plt.show()

def generate_training_report(metrics: Dict, save_dir: str = "./outputs"):
    """Generate training report"""
    
    token_counts = metrics['token_counts']
    rewards = metrics['rewards']
    losses = metrics['losses']
    steps = metrics['steps']
    
    # 检查是否有评估数据
    has_eval_data = 'eval_rewards' in metrics and len(metrics['eval_rewards']) > 0
    
    report = {
        "Training Overview": {
            "Total Training Steps": len(steps),
            "Training Steps Range": f"{min(steps)} - {max(steps)}",
            "Has Evaluation Data": has_eval_data
        },
        "Token Usage Statistics": {
            "Average Token Count": round(np.mean(token_counts), 2),
            "Token Count Median": round(np.median(token_counts), 2),
            "Token Count Std Dev": round(np.std(token_counts), 2),
            "Min Token Count": min(token_counts),
            "Max Token Count": max(token_counts)
        },
        "Training Reward Statistics": {
            "Average Reward": round(np.mean(rewards), 4),
            "Reward Median": round(np.median(rewards), 4),
            "Reward Std Dev": round(np.std(rewards), 4),
            "Min Reward": round(min(rewards), 4),
            "Max Reward": round(max(rewards), 4),
            "Positive Reward Ratio": f"{sum(1 for r in rewards if r > 0) / len(rewards) * 100:.1f}%"
        },
        "Loss Statistics": {
            "Average Loss": round(np.mean(losses), 6),
            "Loss Median": round(np.median(losses), 6),
            "Loss Std Dev": round(np.std(losses), 6),
            "Min Loss": round(min(losses), 6),
            "Max Loss": round(max(losses), 6)
        }
    }
    
    # 如果有评估数据，添加评估统计信息
    if has_eval_data:
        eval_rewards = metrics['eval_rewards']
        eval_token_counts = metrics['eval_token_counts']
        
        report["Evaluation Reward Statistics"] = {
            "Average Eval Reward": round(np.mean(eval_rewards), 4),
            "Eval Reward Median": round(np.median(eval_rewards), 4),
            "Eval Reward Std Dev": round(np.std(eval_rewards), 4),
            "Min Eval Reward": round(min(eval_rewards), 4),
            "Max Eval Reward": round(max(eval_rewards), 4),
            "Positive Eval Reward Ratio": f"{sum(1 for r in eval_rewards if r > 0) / len(eval_rewards) * 100:.1f}%"
        }
        
        report["Evaluation Token Usage Statistics"] = {
            "Average Eval Token Count": round(np.mean(eval_token_counts), 2),
            "Eval Token Count Median": round(np.median(eval_token_counts), 2),
            "Eval Token Count Std Dev": round(np.std(eval_token_counts), 2),
            "Min Eval Token Count": min(eval_token_counts),
            "Max Eval Token Count": max(eval_token_counts)
        }
        
        # 比较训练和评估奖励
        train_eval_comparison = {
            "Train vs Eval Reward Difference": round(np.mean(rewards) - np.mean(eval_rewards), 4),
            "Train vs Eval Token Difference": round(np.mean(token_counts) - np.mean(eval_token_counts), 2),
            "Eval Better Than Train Ratio": f"{sum(1 for i in range(min(len(rewards), len(eval_rewards))) if eval_rewards[i] > rewards[i]) / min(len(rewards), len(eval_rewards)) * 100:.1f}%"
        }
        report["Training vs Evaluation Comparison"] = train_eval_comparison
    
    # Save report
    report_path = os.path.join(save_dir, "training_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print report
    print("\n" + "="*60)
    print("Training Report")
    print("="*60)
    
    for category, stats in report.items():
        print(f"\n{category}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    print(f"\nDetailed report saved to: {report_path}")

def main():
    """Main function"""
    print("Starting training visualization generation...")
    
    # Load metrics data
    metrics = load_training_metrics()
    
    if metrics is None:
        print("Cannot load training metrics, please ensure training is completed")
        return
    
    # Generate visualization charts
    print("\nGenerating training trend charts...")
    plot_training_metrics(metrics)
    
    print("\nGenerating metrics distribution charts...")
    plot_distribution_analysis(metrics)
    
    print("\nGenerating training report...")
    generate_training_report(metrics)
    
    print("\nVisualization completed!")
    print("Generated files:")
    print("- ./outputs/training_metrics.png")
    print("- ./outputs/metrics_distribution.png") 
    print("- ./outputs/training_report.json")

if __name__ == "__main__":
    main() 