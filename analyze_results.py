import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from os.path import dirname, join


def load_metrics(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['training_history'], data['test_results']


def print_summary(history, test_results):
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    # Training info
    num_epochs = len(history['train_loss'])
    print(f"\nTotal Epochs: {num_epochs}")
    
    # Best training metrics
    best_train_loss = min(history['train_loss'])
    best_train_loss_epoch = np.argmin(history['train_loss']) + 1
    print(f"\nBest Training Loss: {best_train_loss:.4f} (Epoch {best_train_loss_epoch})")
    
    # Best validation metrics
    best_val_loss = min(history['val_loss'])
    best_val_loss_epoch = np.argmin(history['val_loss']) + 1
    print(f"Best Validation Loss: {best_val_loss:.4f} (Epoch {best_val_loss_epoch})")
    
    # Primary metric
    if 'val_mean_auc' in history:
        best_auc = max(history['val_mean_auc'])
        best_auc_epoch = np.argmax(history['val_mean_auc']) + 1
        print(f"Best Validation AUC: {best_auc:.4f} (Epoch {best_auc_epoch})")
    elif 'val_accuracy' in history:
        best_acc = max(history['val_accuracy'])
        best_acc_epoch = np.argmax(history['val_accuracy']) + 1
        print(f"Best Validation Accuracy: {best_acc:.4f} (Epoch {best_acc_epoch})")
    
    # Test results
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    for key, value in test_results.items():
        if key not in ['auc_per_label'] and not isinstance(value, list):
            print(f"{key.replace('_', ' ').title():.<30} {value:.4f}")
    
    # Per-label results (if available)
    if 'auc_per_label' in test_results and test_results['auc_per_label']:
        print("\n" + "-"*70)
        print("PER-LABEL AUC")
        print("-"*70)
        label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        for i, (name, auc) in enumerate(zip(label_names, test_results['auc_per_label'])):
            print(f"{name:.<30} {auc:.4f}")
    
    print("\n" + "="*70 + "\n")


def analyze_convergence(history):
    train_loss = np.array(history['train_loss'])
    val_loss = np.array(history['val_loss'])
    
    # Find convergence point (when validation loss stabilizes)
    val_loss_diff = np.abs(np.diff(val_loss))
    convergence_threshold = 0.01
    
    for i, diff in enumerate(val_loss_diff):
        if i > 5 and diff < convergence_threshold:  # After at least 5 epochs
            if np.mean(val_loss_diff[i:min(i+5, len(val_loss_diff))]) < convergence_threshold:
                print(f"✓ Model converged around epoch {i+1}")
                print(f"  Validation loss stabilized at {val_loss[i]:.4f}")
                break
    else:
        print("⚠ Model may not have fully converged")
        print(f"  Validation loss at end: {val_loss[-1]:.4f}")
    
    # Overfitting analysis
    gap = val_loss[-1] - train_loss[-1]
    if gap > 0.1:
        print(f"⚠ Potential overfitting detected (gap: {gap:.4f})")
    else:
        print(f"✓ Minimal overfitting (gap: {gap:.4f})")


def compare_label_difficulty(test_results):
    if 'auc_per_label' not in test_results or not test_results['auc_per_label']:
        print("No per-label data available")
        return
    
    label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    aucs = test_results['auc_per_label']
    
    # Sort by difficulty
    sorted_labels = sorted(zip(label_names, aucs), key=lambda x: x[1])
    
    print("\n" + "="*70)
    print("LABEL DIFFICULTY RANKING (Easiest to Hardest)")
    print("="*70)
    for i, (name, auc) in enumerate(reversed(sorted_labels), 1):
        difficulty = "Easy" if auc > 0.75 else "Medium" if auc > 0.65 else "Hard"
        print(f"{i}. {name:.<30} AUC: {auc:.4f} ({difficulty})")


def create_learning_rate_analysis(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss improvement
    train_loss_improvement = np.diff(history['train_loss'])
    val_loss_improvement = np.diff(history['val_loss'])
    
    ax1.plot(epochs[1:], -train_loss_improvement, label='Train Loss Improvement')
    ax1.plot(epochs[1:], -val_loss_improvement, label='Val Loss Improvement')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss Improvement (Higher = Better)')
    ax1.set_title('Learning Progress per Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative improvement
    ax2.plot(epochs, np.cumsum(-train_loss_improvement), label='Train')
    ax2.plot(epochs, np.cumsum(-val_loss_improvement), label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cumulative Loss Improvement')
    ax2.set_title('Cumulative Learning Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved learning analysis to {save_path}")
    
    return fig


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <path_to_metrics.json>")
        print("\nExample:")
        print("  python analyze_results.py logs/TextGNN_jigsaw_*/plots/metrics.json")
        sys.exit(1)
    
    metrics_file = sys.argv[1]
    print(f"Loading: {metrics_file}")
    
    history, test_results = load_metrics(metrics_file)
    
    # Print summary
    print_summary(history, test_results)
    
    # Convergence analysis
    print("\n" + "="*70)
    print("CONVERGENCE ANALYSIS")
    print("="*70)
    analyze_convergence(history)
    
    # Label difficulty
    compare_label_difficulty(test_results)
    
    # Create additional analysis plot
    output_dir = dirname(metrics_file)
    create_learning_rate_analysis(history, join(output_dir, 'learning_progress.png'))
    
    print("\n" + "="*70)
    print("✓ Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
