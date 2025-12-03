from eval import eval, MovingAverage
from config import FLAGS
from model_factory import create_model

from pprint import pprint
import time
import torch
import gc


def train(train_data, val_data, saver):
    print("  → Initializing node features...")
    train_data.init_node_feats(FLAGS.init_type, FLAGS.device)
    val_data.init_node_feats(FLAGS.init_type, FLAGS.device)
    print(f"  ✓ Feature initialization: {FLAGS.init_type}")
    
    print("  → Creating model...")
    model = create_model(train_data)
    model = model.to(FLAGS.device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model created with {pytorch_total_params:,} parameters")
    
    print(f"  → Setting up training (lr={FLAGS.lr}, epochs={FLAGS.num_epochs})...")
    moving_avg = MovingAverage(FLAGS.validation_window_size, FLAGS.validation_metric != 'loss')
    pyg_graph_train = train_data.get_pyg_graph(FLAGS.device)
    # When using separate graphs, val_data has its own graph
    pyg_graph_val = val_data.get_pyg_graph(FLAGS.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, )
    
    # Initialize training history for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
    }
    
    print(f"\n  {'='*56}")
    print(f"  Starting training for {FLAGS.num_epochs} epochs")
    print(f"  {'='*56}")

    for epoch in range(FLAGS.num_epochs):
        t = time.time()
        model.train()
        model.zero_grad()
        loss, preds_train = model(pyg_graph_train, train_data)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        
        # Store training loss
        history['train_loss'].append(loss)
        
        # Clean up gradients and cached tensors
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        with torch.no_grad():
            # Evaluate on training set (for metrics)
            eval_res_train = eval(preds_train, train_data)
            
            # Evaluate on validation set
            val_loss, preds_val = model(pyg_graph_val, val_data)
            val_loss = val_loss.item()
            eval_res_val = eval(preds_val, val_data)
            
            # Store validation loss
            history['val_loss'].append(val_loss)
            
            # Store all metrics
            for key, value in eval_res_train.items():
                if key not in ['y_true', 'y_pred', 'y_pred_proba']:
                    history_key = f'train_{key}'
                    if history_key not in history:
                        history[history_key] = []
                    history[history_key].append(float(value) if not isinstance(value, list) else value)
            
            for key, value in eval_res_val.items():
                if key not in ['y_true', 'y_pred', 'y_pred_proba']:
                    history_key = f'val_{key}'
                    if history_key not in history:
                        history[history_key] = []
                    history[history_key].append(float(value) if not isinstance(value, list) else value)
            
            # Compact epoch summary
            metric_val = eval_res_val[FLAGS.validation_metric]
            is_best = len(moving_avg.results) == 0 or moving_avg.best_result(metric_val)
            best_marker = " 🌟 NEW BEST" if is_best else ""
            
            print(f"  Epoch {epoch+1:03d}/{FLAGS.num_epochs}: "
                  f"Train Loss={loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"{FLAGS.validation_metric}={metric_val:.4f}, "
                  f"Time={time.time()-t:.2f}s{best_marker}")
            
            eval_res_val["loss"] = val_loss

            if is_best:
                saver.save_trained_model(model, epoch + 1)
                print(f"    → Model saved at epoch {epoch+1}")
            
            moving_avg.add_to_moving_avg(metric_val)
            
            # Periodic garbage collection every 10 epochs
            if (epoch + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # if moving_avg.stop():
            #     print(f"\n  ⚠ Early stopping triggered at epoch {epoch+1}")
            #     break
    
    print(f"\n  {'='*56}")
    print("  Training complete! Loading best model...")
    print(f"  {'='*56}")
    best_model = saver.load_trained_model(train_data)
    return best_model, model, history