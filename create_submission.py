"""
Generate Kaggle submission file for Jigsaw Toxic Comment Classification Challenge
"""
import pandas as pd
import numpy as np
from scipy.special import expit
from config import FLAGS
from utils import get_data_path, get_corpus_path
from os.path import join
import torch

def create_submission(model, test_data):
    """
    Generate predictions for test set and create submission CSV.
    
    Args:
        model: Trained TextGNN model
        test_data: TextDataset object with test samples
    
    Returns:
        Path to submission file
    """
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        pyg_graph = test_data.get_pyg_graph(FLAGS.device)
        _, logits = model(pyg_graph, test_data)
    
    # Convert logits to probabilities using sigmoid
    probabilities = expit(logits)
    
    # Load test IDs
    test_ids_path = join(get_corpus_path(), 'jigsaw_test_ids.txt')
    with open(test_ids_path, 'r') as f:
        test_ids = [line.strip() for line in f]
    
    # Ensure we have the right number of predictions
    assert len(test_ids) == len(probabilities), \
        f"Mismatch: {len(test_ids)} IDs vs {len(probabilities)} predictions"
    
    # Create submission dataframe
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    submission_df = pd.DataFrame({
        'id': test_ids,
        **{col: probabilities[:, i] for i, col in enumerate(label_columns)}
    })
    
    # Save submission file
    submission_path = join(get_data_path(), 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Submission file created: {submission_path}")
    print(f"{'='*60}")
    print(f"\nFirst 5 rows:")
    print(submission_df.head())
    print(f"\nSubmission shape: {submission_df.shape}")
    print(f"\nPrediction statistics:")
    for col in label_columns:
        print(f"  {col}: mean={submission_df[col].mean():.4f}, "
              f"std={submission_df[col].std():.4f}, "
              f"max={submission_df[col].max():.4f}")
    
    return submission_path


if __name__ == "__main__":
    print("To generate submission:")
    print("1. Train your model using: python main.py")
    print("2. Then run this script with the trained model")
    print("\nExample usage in code:")
    print("  from create_submission import create_submission")
    print("  submission_path = create_submission(trained_model, test_dataset)")
