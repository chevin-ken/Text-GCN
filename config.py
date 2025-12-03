from utils import get_host, get_user

import argparse
import torch

parser = argparse.ArgumentParser()

"""
Most Relevant
"""

debug = False
gpu = -1

""" 
dataset:
 sentiment suffix for twitter means the negative classes of the original dataset are combined and the other classes are combined for sentiment analysis
 presplit suffix means training and test are predetermined in [dataset]_labels.txt
 small suffix means a very small dataset used for debugging
"""
# dataset = 'twitter_asian_prejudice'
# dataset = 'twitter_asian_prejudice_sentiment'
# dataset = 'r8_presplit'
# dataset = 'ag_presplit'
# dataset = 'twitter_asian_prejudice_small'
dataset = 'jigsaw'

# Configure dataset-specific parameters
if 'twitter_asian_prejudice' in dataset:
    if 'sentiment' in dataset:
        num_labels = 2
    else:
        num_labels = 4
    multi_label = False
elif 'ag' in dataset:
    num_labels = 4
    multi_label = False
elif 'r8' in dataset:
    num_labels = 8
    multi_label = False
elif 'jigsaw' in dataset:
    num_labels = 6  # toxic, severe_toxic, obscene, threat, insult, identity_hate
    multi_label = True
else:
    raise ValueError(f"Unknown dataset: {dataset}")

parser.add_argument('--dataset', default=dataset)
parser.add_argument('--random_seed', default=3)
parser.add_argument('--multi_label', default=multi_label)


"""
Model. Pt1
"""

model = "text_gcn"

model_params = {}
parser.add_argument('--use_edge_weights', default=False)
parser.add_argument('--init_type', default='one_hot_init')
if model == 'text_gcn':
    n = '--model'
    # Use sigmoid for multi-label, softmax for single-label
    pred_type = 'sigmoid' if multi_label else 'softmax'
    node_embd_type = 'gcn'
    layer_dim_list = [200, num_labels]
    num_layers = len(layer_dim_list)
    class_weights = True
    dropout = True
    s = 'TextGNN:pred_type={},node_embd_type={},num_layers={},layer_dim_list={},act={},' \
        'dropout={},class_weights={}'.format(
        pred_type, node_embd_type, num_layers, "_".join([str(i) for i in layer_dim_list]), 'relu', dropout, class_weights
    )
    model_params = {
        'pred_type': pred_type,
        'node_embd':  node_embd_type,
        'layer_dims': layer_dim_list,
        'class_weights': class_weights,
        'dropout': dropout
    }
    parser.add_argument(n, default=s)
else:
    raise NotImplementedError

print("{}: {}\n".format(model, model_params))

"""
Sampling
"""
# Reduced from 10 to 5 for memory efficiency with large datasets
word_window_size = 5
parser.add_argument('--word_window_size', default=word_window_size)

# Minimum word frequency to include in vocabulary (filters rare words)
# Higher value = smaller vocabulary = less memory
min_word_freq = 5  # Only include words appearing at least 5 times
parser.add_argument('--min_word_freq', type=int, default=min_word_freq)

# Use separate graphs for train/val/test instead of a single unified graph
# By default, use unified graph (original behavior)
# Add --use_separate_graphs flag to enable separate graphs mode
parser.add_argument('--use_separate_graphs', action='store_true', 
                    help='Build separate graphs for train/val/test instead of unified graph')

validation_window_size = 10

"""
Validation
"""
parser.add_argument("--validation_window_size", default=validation_window_size)
# Default validation metric depends on task type
default_val_metric = "mean_auc" if multi_label else "accuracy"
parser.add_argument("--validation_metric", default=default_val_metric,
                    choices=["f1_weighted", "accuracy", "loss", "mean_auc", "hamming_loss"])

use_best_val_model_for_inference = True
parser.add_argument('--use_best_val_model_for_inference', default=use_best_val_model_for_inference)

"""
Evaluation.
"""
tvt_ratio = [0.8, 0.1, 0.1]
parser.add_argument('--tvt_ratio', default=tvt_ratio)
parser.add_argument('--tvt_list', default=["train", "test", "val"])


"""
Optimization.
"""

lr = 2e-2
parser.add_argument('--lr', type=float, default=lr)


device = str('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1
             else 'cpu')
parser.add_argument('--device', default=device)


num_epochs = 2
num_epochs = 2 if debug else num_epochs
parser.add_argument('--num_epochs', type=int, default=num_epochs)



"""
Other info.
"""
parser.add_argument('--user', default=get_user())

parser.add_argument('--hostname', default=get_host())

# Use parse_known_args() to support Jupyter/Colab environments
# This ignores unknown arguments (like Jupyter kernel args)
FLAGS, unknown = parser.parse_known_args()



