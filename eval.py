import numpy as np
from sklearn import metrics
from scipy.special import expit  # sigmoid function


def eval(preds, dataset, test=False):
    y_true = dataset.label_inds[dataset.node_ids]
    
    # Check if multi-label or single-label
    is_multi_label = dataset.multi_label if hasattr(dataset, 'multi_label') else False
    
    if test:
        mode_str = "multi-label" if is_multi_label else "single-label"
        print(f"    → Evaluating in {mode_str} mode...")
    
    if is_multi_label:
        # Multi-label classification metrics
        # Apply sigmoid to get probabilities
        y_pred_proba = expit(preds)  # Convert logits to probabilities
        
        # Binary predictions with threshold 0.5
        y_pred_label = (y_pred_proba > 0.5).astype(int)
        
        # Compute multi-label metrics
        results = {}
        
        # Mean AUC across all labels (primary Kaggle metric)
        try:
            auc_scores = []
            for i in range(y_true.shape[1]):
                if len(np.unique(y_true[:, i])) > 1:  # Only compute if label has both classes
                    auc = metrics.roc_auc_score(y_true[:, i], y_pred_proba[:, i])
                    auc_scores.append(auc)
            results["mean_auc"] = np.mean(auc_scores) if auc_scores else 0.0
            results["auc_per_label"] = auc_scores
        except:
            results["mean_auc"] = 0.0
            results["auc_per_label"] = []
        
        # Hamming loss (fraction of wrong labels)
        results["hamming_loss"] = metrics.hamming_loss(y_true, y_pred_label)
        
        # Subset accuracy (exact match ratio)
        results["subset_accuracy"] = metrics.accuracy_score(y_true, y_pred_label)
        
        # Micro/macro averaged metrics
        results["f1_micro"] = metrics.f1_score(y_true, y_pred_label, average='micro', zero_division=0)
        results["f1_macro"] = metrics.f1_score(y_true, y_pred_label, average='macro', zero_division=0)
        results["f1_weighted"] = metrics.f1_score(y_true, y_pred_label, average='weighted', zero_division=0)
        results["precision_micro"] = metrics.precision_score(y_true, y_pred_label, average='micro', zero_division=0)
        results["recall_micro"] = metrics.recall_score(y_true, y_pred_label, average='micro', zero_division=0)
        
        # For compatibility
        results["accuracy"] = results["subset_accuracy"]
        
        if test:
            results["y_true"] = y_true
            results["y_pred"] = y_pred_label
            results["y_pred_proba"] = y_pred_proba
    else:
        # Single-label classification (original code)
        y_pred_label = np.asarray([np.argmax(pred) for pred in preds])
        accuracy = metrics.accuracy_score(y_true, y_pred_label)
        f1_weighted = metrics.f1_score(y_true, y_pred_label, average='weighted')
        f1_macro = metrics.f1_score(y_true, y_pred_label, average='macro')
        f1_micro = metrics.f1_score(y_true, y_pred_label, average='micro')
        precision_weighted = metrics.precision_score(y_true, y_pred_label, average='weighted')
        precision_macro = metrics.precision_score(y_true, y_pred_label, average='macro')
        precision_micro = metrics.precision_score(y_true, y_pred_label, average='micro')
        recall_weighted = metrics.recall_score(y_true, y_pred_label, average='weighted')
        recall_macro = metrics.recall_score(y_true, y_pred_label, average='macro')
        recall_micro = metrics.recall_score(y_true, y_pred_label, average='micro')
        results = {"accuracy": accuracy,
                   "f1_weighted": f1_weighted,
                   "f1_macro": f1_macro,
                   "f1_micro": f1_micro,
                   "precision_weighted": precision_weighted,
                   "precision_macro": precision_macro,
                   "precision_micro": precision_micro,
                   "recall_weighted": recall_weighted,
                   "recall_macro": recall_macro,
                   "recall_micro": recall_micro
                   }
        if test:
            one_hot_true = np.zeros((y_true.size, len(dataset.label_dict)))
            one_hot_true[np.arange(y_true.size), y_true] = 1
            results["y_true"] = one_hot_true
            one_hot_pred = np.zeros((y_true.size, len(dataset.label_dict)))
            one_hot_pred[np.arange(y_pred_label.size),y_pred_label] = 1
            results["y_pred"] = one_hot_pred
    
    return results


class MovingAverage(object):
    def __init__(self, window, want_increase=True):
        self.moving_avg = [float('-inf')] if want_increase else [float('inf')]
        self.want_increase = want_increase
        self.results = []
        self.window = window

    def add_to_moving_avg(self, x):
        self.results.append(x)
        if len(self.results) >= self.window:
            next_val = sum(self.results[-self.window:]) / self.window
            self.moving_avg.append(next_val)

    def best_result(self, x):
        if self.want_increase:
            return (x - 1e-7) > max(self.results)
        else:
            return (x + 1e-7) < min(self.results)

    def stop(self):
        if len(self.moving_avg) < 2:
            return False
        if self.want_increase:
            return (self.moving_avg[-1] + 1e-7) < self.moving_avg[-2]
        else:
            return (self.moving_avg[-2] + 1e-7) < self.moving_avg[-1]