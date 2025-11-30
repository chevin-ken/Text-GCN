from config import FLAGS
from eval import eval
from load_data import load_data
from saver import Saver
from train import train

from pprint import pprint
import torch
import gc


def main():
    print("\n" + "="*60)
    print("STARTING TEXT-GCN TRAINING PIPELINE")
    print("="*60)
    print(f"Dataset: {FLAGS.dataset}")
    print(f"Device: {FLAGS.device}")
    print(f"Multi-label: {FLAGS.multi_label}")
    print(f"Validation metric: {FLAGS.validation_metric}")
    print("="*60 + "\n")
    
    print("[1/5] Initializing saver...")
    saver = Saver()
    
    print("[2/5] Loading data...")
    train_data, val_data, test_data, raw_doc_list = load_data()
    print(f"  ✓ Graph shape: {train_data.graph.shape}")
    print(f"  ✓ Train samples: {len(train_data.node_ids)}")
    print(f"  ✓ Val samples: {len(val_data.node_ids)}")
    print(f"  ✓ Test samples: {len(test_data.node_ids)}")
    print(f"  ✓ Vocabulary size: {len(train_data.vocab)}")
    
    # Memory cleanup after data loading
    gc.collect()
    
    print("\n[3/5] Training model...")
    saved_model, model = train(train_data, val_data, saver)
    
    # Memory cleanup after training
    del saved_model  # We only need the final model
    gc.collect()
    
    print("\n[4/5] Evaluating on test set...")
    with torch.no_grad():
        test_loss_model, preds_model = model(train_data.get_pyg_graph(device=FLAGS.device), test_data)
    
    print("[5/5] Computing final test metrics...")
    eval_res = eval(preds_model, test_data, True)
    y_true = eval_res.pop('y_true')
    y_pred = eval_res.pop('y_pred')
    
    # Clean up predictions after evaluation
    del preds_model
    gc.collect()
    
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    pprint(eval_res)
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
