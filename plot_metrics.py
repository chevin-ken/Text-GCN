import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
import json


def plot_loss_curves(history, save_path=None, title="Training and Validation Loss"):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add min/max annotations
    min_train_idx = np.argmin(history['train_loss'])
    min_val_idx = np.argmin(history['val_loss'])
    ax.plot(min_train_idx + 1, history['train_loss'][min_train_idx], 'b*', markersize=15)
    ax.plot(min_val_idx + 1, history['val_loss'][min_val_idx], 'r*', markersize=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved loss curve to {save_path}")
    
    return fig


def plot_metric_curves(history, metric_name, save_path=None):
    """Plot training and validation metric curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history[f'train_{metric_name}']) + 1)
    ax.plot(epochs, history[f'train_{metric_name}'], 'b-', 
            label=f'Training {metric_name.upper()}', linewidth=2)
    ax.plot(epochs, history[f'val_{metric_name}'], 'r-', 
            label=f'Validation {metric_name.upper()}', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric_name.upper(), fontsize=12)
    ax.set_title(f'{metric_name.upper()} over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add max annotations
    max_train_idx = np.argmax(history[f'train_{metric_name}'])
    max_val_idx = np.argmax(history[f'val_{metric_name}'])
    ax.plot(max_train_idx + 1, history[f'train_{metric_name}'][max_train_idx], 'b*', markersize=15)
    ax.plot(max_val_idx + 1, history[f'val_{metric_name}'][max_val_idx], 'r*', markersize=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved {metric_name} curve to {save_path}")
    
    return fig


def plot_all_metrics(history, save_dir):
    """Plot all available metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n  → Generating training visualizations...")
    
    # 1. Loss curves
    plot_loss_curves(history, join(save_dir, 'loss_curves.png'))
    
    # 2. Primary validation metric
    metrics_to_plot = []
    if 'val_mean_auc' in history:
        metrics_to_plot.append('mean_auc')
    if 'val_accuracy' in history:
        metrics_to_plot.append('accuracy')
    if 'val_f1_weighted' in history:
        metrics_to_plot.append('f1_weighted')
    
    for metric in metrics_to_plot:
        if f'train_{metric}' in history and f'val_{metric}' in history:
            plot_metric_curves(history, metric, join(save_dir, f'{metric}_curves.png'))
    
    # 3. Multi-metric comparison
    if len(metrics_to_plot) > 1:
        plot_multiple_metrics(history, metrics_to_plot, join(save_dir, 'all_metrics_comparison.png'))
    
    # 4. Per-label AUC (if available)
    if 'val_auc_per_label' in history and history['val_auc_per_label']:
        plot_per_label_auc(history, join(save_dir, 'per_label_auc.png'))
    
    print(f"  ✓ All plots saved to {save_dir}")


def plot_multiple_metrics(history, metric_names, save_path):
    """Plot multiple metrics on the same graph."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = range(1, len(history['val_loss']) + 1)
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown']
    
    for i, metric in enumerate(metric_names):
        if f'val_{metric}' in history:
            ax.plot(epochs, history[f'val_{metric}'], 
                   color=colors[i % len(colors)], 
                   label=metric.upper(), linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Validation Metrics Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved multi-metric comparison to {save_path}")
    
    return fig


def plot_per_label_auc(history, save_path):
    """Plot AUC for each label over epochs (for multi-label classification)."""
    # Extract per-label AUC from history
    # history['val_auc_per_label'] is a list of lists
    per_label_history = history['val_auc_per_label']
    
    if not per_label_history or not per_label_history[0]:
        return
    
    num_labels = len(per_label_history[0])
    epochs = range(1, len(per_label_history) + 1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Default label names
    label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    if num_labels > len(label_names):
        label_names = [f'Label {i}' for i in range(num_labels)]
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_labels))
    
    for label_idx in range(num_labels):
        label_auc_history = [epoch_aucs[label_idx] for epoch_aucs in per_label_history]
        ax.plot(epochs, label_auc_history, 
               label=label_names[label_idx], 
               linewidth=2, marker='o', markersize=4, color=colors[label_idx])
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Per-Label AUC over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])  # AUC ranges from 0.5 to 1.0
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved per-label AUC plot to {save_path}")
    
    return fig


def plot_final_test_metrics(test_results, save_path):
    """Create a bar chart of final test metrics."""
    # Extract key metrics for visualization
    metrics_to_show = {}
    
    for key in ['accuracy', 'f1_weighted', 'f1_macro', 'f1_micro', 
                'precision_micro', 'recall_micro', 'mean_auc']:
        if key in test_results:
            value = test_results[key]
            if isinstance(value, (np.ndarray, np.generic)):
                value = float(value)
            metrics_to_show[key] = value
    
    if not metrics_to_show:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = list(metrics_to_show.keys())
    metric_values = list(metrics_to_show.values())
    
    bars = ax.bar(range(len(metric_names)), metric_values, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels([name.replace('_', ' ').title() for name in metric_names], rotation=45, ha='right')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Final Test Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, metric_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved test metrics bar chart to {save_path}")
    
    return fig


def save_metrics_json(history, test_results, save_path):
    """Save all metrics to a JSON file for later analysis."""
    data = {
        'training_history': history,
        'test_results': {}
    }
    
    # Convert numpy types to Python types for JSON serialization
    for key, value in test_results.items():
        if isinstance(value, np.ndarray):
            data['test_results'][key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            data['test_results'][key] = float(value)
        elif key not in ['y_true', 'y_pred', 'y_pred_proba']:  # Skip large arrays
            data['test_results'][key] = value
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  ✓ Saved metrics JSON to {save_path}")


def create_training_summary_plot(history, test_results, save_path):
    """Create a comprehensive summary plot with multiple subplots."""
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Loss curves
    ax1 = plt.subplot(2, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Primary metric curves
    ax2 = plt.subplot(2, 3, 2)
    if 'val_mean_auc' in history:
        metric_key = 'mean_auc'
        metric_label = 'Mean AUC'
    elif 'val_accuracy' in history:
        metric_key = 'accuracy'
        metric_label = 'Accuracy'
    else:
        metric_key = 'f1_weighted'
        metric_label = 'F1 Weighted'
    
    if f'train_{metric_key}' in history:
        ax2.plot(epochs, history[f'train_{metric_key}'], 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, history[f'val_{metric_key}'], 'r-', label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(metric_label)
    ax2.set_title(f'{metric_label} over Epochs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Final test metrics bar chart
    ax3 = plt.subplot(2, 3, 3)
    test_metrics = {}
    for key in ['accuracy', 'f1_weighted', 'mean_auc']:
        if key in test_results:
            value = test_results[key]
            if isinstance(value, (np.ndarray, np.generic)):
                value = float(value)
            test_metrics[key] = value
    
    if test_metrics:
        bars = ax3.bar(range(len(test_metrics)), list(test_metrics.values()), color='steelblue')
        ax3.set_xticks(range(len(test_metrics)))
        ax3.set_xticklabels([k.replace('_', ' ').title() for k in test_metrics.keys()], rotation=45, ha='right')
        ax3.set_ylabel('Score')
        ax3.set_title('Final Test Metrics')
        ax3.set_ylim([0, 1.0])
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, test_metrics.values()):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. F1 scores comparison (if available)
    ax4 = plt.subplot(2, 3, 4)
    if 'val_f1_weighted' in history:
        ax4.plot(epochs, history['val_f1_weighted'], label='F1 Weighted', linewidth=2)
    if 'val_f1_macro' in history:
        ax4.plot(epochs, history['val_f1_macro'], label='F1 Macro', linewidth=2)
    if 'val_f1_micro' in history:
        ax4.plot(epochs, history['val_f1_micro'], label='F1 Micro', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1 Score')
    ax4.set_title('F1 Scores Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Per-label AUC (if multi-label)
    ax5 = plt.subplot(2, 3, 5)
    if 'val_auc_per_label' in history and history['val_auc_per_label']:
        per_label = history['val_auc_per_label']
        num_labels = len(per_label[0]) if per_label[0] else 0
        label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        for label_idx in range(min(num_labels, len(label_names))):
            label_history = [epoch[label_idx] for epoch in per_label]
            ax5.plot(epochs, label_history, label=label_names[label_idx], linewidth=1.5)
        
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('AUC')
        ax5.set_title('Per-Label AUC')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0.5, 1.0])
    
    # 6. Training summary text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = "Training Summary\n" + "="*30 + "\n\n"
    summary_text += f"Total Epochs: {len(epochs)}\n\n"
    summary_text += f"Best Val Loss: {min(history['val_loss']):.4f}\n"
    summary_text += f"  (Epoch {np.argmin(history['val_loss']) + 1})\n\n"
    
    if f'val_{metric_key}' in history:
        best_metric = max(history[f'val_{metric_key}'])
        summary_text += f"Best Val {metric_label}: {best_metric:.4f}\n"
        summary_text += f"  (Epoch {np.argmax(history[f'val_{metric_key}']) + 1})\n\n"
    
    summary_text += "\nFinal Test Results:\n"
    for key, value in test_metrics.items():
        summary_text += f"  {key.replace('_', ' ').title()}: {value:.4f}\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved comprehensive summary plot to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Example usage
    print("Use this module by importing it in your training script")
    print("Example: from plot_metrics import plot_all_metrics, save_metrics_json")
