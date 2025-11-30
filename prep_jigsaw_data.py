import pandas as pd
import numpy as np
from os.path import join
from utils import get_data_path

def convert_jigsaw_to_textgcn_format():
    """
    Convert Jigsaw Toxic Comment CSV files to TextGCN format.
    
    Creates:
    - data/corpus/jigsaw_sentences.txt: One comment per line
    - data/corpus/jigsaw_labels.txt: Multi-label format (comma-separated binary values)
    """
    # Read training data
    train_df = pd.read_csv(join(get_data_path(), 'train.csv'))
    
    print(f"Loaded {len(train_df)} training samples")
    
    # Extract comment text
    comments = train_df['comment_text'].values
    
    # Extract labels (6 binary columns)
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    labels = train_df[label_columns].values  # Shape: (n_samples, 6)
    
    # Print label statistics
    print("\nLabel distribution:")
    for i, col in enumerate(label_columns):
        pos_count = labels[:, i].sum()
        print(f"  {col}: {pos_count} ({pos_count/len(labels)*100:.2f}%)")
    
    # Calculate samples with at least one label
    has_any_label = (labels.sum(axis=1) > 0).sum()
    print(f"\nSamples with at least one toxic label: {has_any_label} ({has_any_label/len(labels)*100:.2f}%)")
    
    # Save sentences
    sentences_path = join(get_data_path(), 'corpus', 'jigsaw_sentences.txt')
    with open(sentences_path, 'w', encoding='utf-8') as f:
        for comment in comments:
            # Replace newlines and tabs with spaces to keep one comment per line
            cleaned = str(comment).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            f.write(cleaned + '\n')
    print(f"\nSaved sentences to: {sentences_path}")
    
    # Save labels in comma-separated binary format
    # Format: "0,1,0,0,1,0" for each row
    labels_path = join(get_data_path(), 'corpus', 'jigsaw_labels.txt')
    with open(labels_path, 'w', encoding='utf-8') as f:
        for label_row in labels:
            label_str = ','.join(map(str, label_row.astype(int)))
            f.write(label_str + '\n')
    print(f"Saved labels to: {labels_path}")
    
    # Also process test data (for future submission)
    test_df = pd.read_csv(join(get_data_path(), 'test.csv'))
    print(f"\nLoaded {len(test_df)} test samples")
    
    test_sentences_path = join(get_data_path(), 'corpus', 'jigsaw_test_sentences.txt')
    with open(test_sentences_path, 'w', encoding='utf-8') as f:
        for comment in test_df['comment_text'].values:
            cleaned = str(comment).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            f.write(cleaned + '\n')
    print(f"Saved test sentences to: {test_sentences_path}")
    
    # Save test IDs for submission
    test_ids_path = join(get_data_path(), 'corpus', 'jigsaw_test_ids.txt')
    test_df['id'].to_csv(test_ids_path, index=False, header=False)
    print(f"Saved test IDs to: {test_ids_path}")
    
    print("\n" + "="*60)
    print("Data conversion complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python prep_data.py (to clean the sentences)")
    print("2. Update config.py to set dataset = 'jigsaw'")
    print("3. Run: python main.py (to train the model)")

if __name__ == "__main__":
    convert_jigsaw_to_textgcn_format()
