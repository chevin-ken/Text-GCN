from config import FLAGS
from collections import Counter
from model_text_gnn import TextGNN
from utils import parse_as_int_list
import torch

def create_model(dataset):
    sp = vars(FLAGS)["model"].split(':')
    name = sp[0]
    layer_info = {}
    if len(sp) > 1:
        assert (len(sp) == 2)
        for spec in sp[1].split(','):
            ssp = spec.split('=')
            layer_info[ssp[0]] = '='.join(ssp[1:])  # could have '=' in layer_info
    if name in model_ctors:
        return model_ctors[name](layer_info, dataset)
    else:
        raise ValueError("Model not implemented {}".format(name))


def create_text_gnn(layer_info, dataset):
    print(f"    → Model type: TextGNN")
    print(f"    → Prediction type: {layer_info['pred_type']}")
    print(f"    → Node embedding: {layer_info['node_embd_type']}")
    
    lyr_dims = parse_as_int_list(layer_info["layer_dim_list"])
    lyr_dims = [dataset.node_feats.shape[1]] + lyr_dims
    print(f"    → Layer dimensions: {lyr_dims}")
    
    # Determine number of labels
    if hasattr(dataset, 'multi_label') and dataset.multi_label:
        num_labels = dataset.label_inds.shape[1]  # Multi-label: use second dimension
        print(f"    → Multi-label mode: {num_labels} labels")
    else:
        num_labels = len(dataset.label_dict)  # Single-label: use label dict
        print(f"    → Single-label mode: {num_labels} classes")
    
    weights = None
    if layer_info["class_weights"].lower() == "true":
        print(f"    → Computing class weights...")
        if hasattr(dataset, 'multi_label') and dataset.multi_label:
            # For multi-label: compute pos_weight for each label
            import numpy as np
            labels = dataset.label_inds[dataset.node_ids]
            pos_counts = labels.sum(axis=0)
            neg_counts = len(labels) - pos_counts
            weights = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float, device=FLAGS.device)
            print(f"      ✓ Computed per-label pos_weights (shape: {weights.shape})")
        else:
            # For single-label: compute per-class weights
            counts = Counter(dataset.label_inds[dataset.node_ids])
            weights = len(counts) * [0]
            min_weight = min(counts.values())
            for k, v in counts.items():
                weights[k] = min_weight / float(v)
            weights = torch.tensor(weights, device=FLAGS.device)
            print(f"      ✓ Computed class weights: {weights.tolist()}")

    return TextGNN(
        pred_type=layer_info["pred_type"],
        node_embd_type=layer_info["node_embd_type"],
        num_layers=int(layer_info["num_layers"]),
        layer_dim_list=lyr_dims,
        act=layer_info["act"],
        bn=False,
        num_labels=num_labels,
        class_weights=weights,
        dropout=layer_info["dropout"]
    )


model_ctors = {
    'TextGNN': create_text_gnn,
}
