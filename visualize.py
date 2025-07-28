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
    
    # Smooth data
    smoothed_tokens = smooth_curve(token_counts)
    smoothed_rewards = smooth_curve(rewards)
    smoothed_losses = smooth_curve(losses)
    
    # Create charts
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
    
    # 2. Reward Trend
    ax2 = axes[0, 1]
    ax2.plot(steps, rewards, alpha=0.3, color='green', label='Raw Data')
    ax2.plot(steps, smoothed_rewards, color='green', linewidth=2, label='Smoothed Trend')
    ax2.set_title('Reward Trend', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Average Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistical information
    mean_reward = np.mean(rewards)
    ax2.axhline(y=mean_reward, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_reward:.3f}')
    ax2.legend()
    
    # 3. Loss Trend
    ax3 = axes[1, 0]
    ax3.plot(steps, losses, alpha=0.3, color='orange', label='Raw Data')
    ax3.plot(steps, smoothed_losses, color='orange', linewidth=2, label='Smoothed Trend')
    ax3.set_title('Loss Trend', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Average Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Comprehensive Metrics Comparison
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
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training Metrics Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Token Count Distribution
    ax1 = axes[0]
    ax1.hist(token_counts, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.mean(token_counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(token_counts):.1f}')
    ax1.axvline(np.median(token_counts), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(token_counts):.1f}')
    ax1.set_title('Token Count Distribution')
    ax1.set_xlabel('Token Count')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Reward Distribution
    ax2 = axes[1]
    ax2.hist(rewards, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.3f}')
    ax2.axvline(np.median(rewards), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.3f}')
    ax2.set_title('Reward Distribution')
    ax2.set_xlabel('Reward Value')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Loss Distribution
    ax3 = axes[2]
    ax3.hist(losses, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(np.mean(losses), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(losses):.4f}')
    ax3.axvline(np.median(losses), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(losses):.4f}')
    ax3.set_title('Loss Distribution')
    ax3.set_xlabel('Loss Value')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
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
    
    report = {
        "Training Overview": {
            "Total Training Steps": len(steps),
            "Training Steps Range": f"{min(steps)} - {max(steps)}"
        },
        "Token Usage Statistics": {
            "Average Token Count": round(np.mean(token_counts), 2),
            "Token Count Median": round(np.median(token_counts), 2),
            "Token Count Std Dev": round(np.std(token_counts), 2),
            "Min Token Count": min(token_counts),
            "Max Token Count": max(token_counts)
        },
        "Reward Statistics": {
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