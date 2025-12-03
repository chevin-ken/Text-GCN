import networkx as nx
import scipy.sparse as sp
from math import log
from dataset import TextDataset
from collections import defaultdict
import pandas as pd
from utils import get_corpus_path
from os.path import join, exists
from tqdm import tqdm
import gc
import numpy as np
from rank_bm25 import BM25Okapi


def build_text_graph_dataset(dataset, window_size=None, use_sentiment=False, ngram_range=(1, 3)):
    # Auto-adjust window size based on dataset
    if window_size is None:
        window_size = 5 if 'jigsaw' in dataset else 20
    
    print(f"      → Dataset: {dataset}, Using BM25 + Character N-grams")
    if "small" in dataset or "presplit" in dataset or 'sentiment' in dataset:
        dataset_name = "_".join(dataset.split("_")[:-1])
    else:
        dataset_name = dataset
    clean_text_path = join(get_corpus_path(), dataset_name + '_sentences_clean.txt')
    labels_path = join(get_corpus_path(), dataset_name + '_labels.txt')
    
    print(f"      → Loading documents from: {dataset_name}_sentences_clean.txt")
    doc_list = []
    f = open(clean_text_path, 'rb')
    for line in f.readlines():
        doc_list.append(line.strip().decode())
    f.close()
    print(f"      ✓ Loaded {len(doc_list)} documents")
    
    # Handle multi-label format for Jigsaw dataset
    if 'jigsaw' in dataset:
        # Labels are comma-separated binary values: "0,1,0,0,1,0"
        labels_list = []
        with open(labels_path, 'r') as f:
            for line in f:
                label_row = [int(x) for x in line.strip().split(',')]
                labels_list.append(label_row)
        split_dict = None
    else:
        # Original single-label format
        labels = pd.read_csv(labels_path, header=None, sep='\t')
        assert len(labels) == len(doc_list)
        if 'presplit' not in dataset:
            labels_list = labels.iloc[0:, 0].tolist()
            split_dict = None
        else:
            labels_list = labels.iloc[0:, 2].tolist()
            split = labels.iloc[0:, 1].tolist()
            split_dict = {}
            for i, v in enumerate(split):
                split_dict[i] = v
    
    assert len(labels_list) == len(doc_list), f"Labels ({len(labels_list)}) and documents ({len(doc_list)}) count mismatch"
    print(f"      ✓ Loaded {len(labels_list)} labels")
    
    if "small" in dataset:
        doc_list = doc_list[:200]
        labels_list = labels_list[:200]
        print(f"      → Using small subset: {len(doc_list)} documents")

    print("      → Building vocabulary...")
    word_freq = get_vocab(doc_list)
    original_vocab_size = len(word_freq)
    
    # Filter vocabulary by minimum frequency to reduce memory
    from config import FLAGS
    min_freq = FLAGS.min_word_freq if hasattr(FLAGS, 'min_word_freq') else 5
    word_freq = {word: freq for word, freq in word_freq.items() if freq >= min_freq}
    
    vocab = list(word_freq.keys())
    print(f"      ✓ Vocabulary: {len(vocab)} words (filtered from {original_vocab_size}, min_freq={min_freq})")
    
    if not exists(join(get_corpus_path(), dataset + '_vocab.txt')):
        vocab_str = '\n'.join(vocab)
        f = open(join(get_corpus_path(), dataset + '_vocab.txt'), 'w')
        f.write(vocab_str)
        f.close()
    
    print("      → Building word-document edges...")
    vocab_set = set(vocab)  # For fast lookup
    words_in_docs, word_doc_freq = build_word_doc_edges(doc_list, vocab_set)
    word_id_map = {word: i for i, word in enumerate(vocab)}
    
    # Clean up words_in_docs as it's no longer needed
    del words_in_docs
    gc.collect()

    print("      → Building graph edges (BM25 + Character N-grams)...")
    sparse_graph = build_bm25_chargram_edges(doc_list, word_id_map, vocab, word_doc_freq)
    print(f"      ✓ Graph built: {sparse_graph.shape[0]} nodes, {sparse_graph.nnz} edges")
    
    # Clean up large intermediate structures
    del word_doc_freq
    gc.collect()
    print("      ✓ Memory cleanup after graph construction")
    
    docs_dict = {i: doc for i, doc in enumerate(doc_list)}
    return TextDataset(dataset, sparse_graph, labels_list, vocab, word_id_map, docs_dict, None,
                       train_test_split=split_dict)


def build_bm25_chargram_edges(doc_list, word_id_map, vocab, word_doc_freq):
    """
    Build graph edges using:
    1. BM25 for document-word edges (better than TF-IDF for short text)
    2. Character n-grams for word-word edges (handles obfuscation: f**k, sh1t)
    
    Args:
        doc_list: List of documents (strings)
        word_id_map: Mapping from words to IDs
        vocab: List of vocabulary words
        word_doc_freq: Document frequency for each word
    """
    
    def get_char_ngrams(word, n=3):
        """Extract character n-grams from word"""
        if len(word) < n:
            return [word]
        return [word[i:i+n] for i in range(len(word) - n + 1)]
    
    def char_ngram_similarity(word1, word2, n=3):
        """
        Jaccard similarity based on character n-grams
        Handles: "fuck" vs "f**k" vs "f*ck" vs "fuk"
        """
        ngrams1 = set(get_char_ngrams(word1, n))
        ngrams2 = set(get_char_ngrams(word2, n))
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0
    
    # Step 1: Build word-word edges with character n-gram similarity
    print("      → Building word-word edges (character 3-grams)...")
    row = []
    col = []
    weight = []
    num_docs = len(doc_list)
    
    similarity_threshold = 0.4  # Lower threshold since char similarity is more strict
    
    # Pre-compute character n-grams for all vocab words
    word_chargrams = {}
    for word in vocab:
        word_chargrams[word] = set(get_char_ngrams(word, 3))
    
    # Build similarity edges
    word_pair_count = 0
    for i, word_i in enumerate(tqdm(vocab, desc="Character n-gram similarities")):
        word_i_id = word_id_map[word_i]
        ngrams_i = word_chargrams[word_i]
        
        # Only compare with subsequent words to avoid duplicates
        for j in range(i + 1, len(vocab)):
            word_j = vocab[j]
            word_j_id = word_id_map[word_j]
            ngrams_j = word_chargrams[word_j]
            
            # Jaccard similarity
            intersection = len(ngrams_i & ngrams_j)
            union = len(ngrams_i | ngrams_j)
            
            if union > 0:
                similarity = intersection / union
                
                if similarity > similarity_threshold:
                    # Add both directions (undirected graph)
                    row.append(num_docs + word_i_id)
                    col.append(num_docs + word_j_id)
                    weight.append(similarity)
                    
                    row.append(num_docs + word_j_id)
                    col.append(num_docs + word_i_id)
                    weight.append(similarity)
                    
                    word_pair_count += 1
    
    print(f"      ✓ Created {word_pair_count} word-word edge pairs ({len(weight)} directed edges)")
    
    # Clean up
    del word_chargrams
    gc.collect()
    
    # Step 3: Build document-word edges with BM25 (OPTIMIZED)
    print("      → Building document-word edges (BM25)...")
    
    # BM25 parameters
    k1 = 1.5  # Term frequency saturation
    b = 0.75  # Length normalization
    
    # Pre-compute IDF for all words in vocabulary (vectorized)
    print("      → Pre-computing IDF values...")
    num_docs_total = len(doc_list)
    idf_cache = {}
    for word in vocab:
        doc_freq = word_doc_freq[word]
        idf_cache[word] = log((num_docs_total - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
    
    # Pre-compute document lengths and average
    print("      → Computing document statistics...")
    doc_lengths = []
    doc_term_freqs = []  # List of Counter objects
    
    from collections import Counter
    for doc in doc_list:
        tokens = [w for w in doc.split() if w in word_id_map]
        doc_lengths.append(len(tokens))
        doc_term_freqs.append(Counter(tokens))
    
    avg_doc_len = sum(doc_lengths) / len(doc_lengths)
    
    # Build edges (now very fast!)
    print("      → Building BM25 edges...")
    for i in tqdm(range(len(doc_list)), desc="BM25 doc-word edges"):
        doc_len = doc_lengths[i]
        term_freq_dict = doc_term_freqs[i]
        
        # Length normalization factor (computed once per doc)
        len_norm = k1 * (1 - b + b * doc_len / avg_doc_len)
        
        # Process each unique word in document
        for word, term_freq in term_freq_dict.items():
            word_id = word_id_map[word]
            
            # BM25 formula with pre-computed IDF
            idf = idf_cache[word]
            bm25_score = idf * (term_freq * (k1 + 1)) / (term_freq + len_norm)
            
            if bm25_score > 0:
                row.append(i)
                col.append(num_docs + word_id)
                weight.append(bm25_score)
    
    doc_word_edge_count = len(weight) - word_pair_count * 2
    print(f"      ✓ Created {doc_word_edge_count} document-word edges")
    
    # Clean up
    del doc_lengths, doc_term_freqs, idf_cache
    gc.collect()

    # Step 4: Build sparse adjacency matrix
    number_nodes = num_docs + len(vocab)
    adj_mat = sp.csr_matrix((weight, (row, col)), shape=(number_nodes, number_nodes))
    
    # Clean up
    del row, col, weight
    gc.collect()
    
    # Make symmetric (undirected graph)
    adj = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(adj_mat.T > adj_mat)
    
    # Clean up
    del adj_mat
    gc.collect()
    
    return adj


def get_vocab(text_list):
    word_freq = defaultdict(int)
    for doc_words in text_list:
        words = doc_words.split()
        for word in words:
            word_freq[word] += 1
    return word_freq


def build_word_doc_edges(doc_list, vocab_set=None):
    # build all docs that a word is contained in
    # Only track words in vocab_set (if provided) to save memory
    words_in_docs = defaultdict(set)
    for i, doc_words in enumerate(doc_list):
        words = doc_words.split()
        for word in words:
            # Skip words not in vocabulary (filtered out)
            if vocab_set is not None and word not in vocab_set:
                continue
            words_in_docs[word].add(i)

    word_doc_freq = {}
    for word, doc_list in words_in_docs.items():
        word_doc_freq[word] = len(doc_list)

    return words_in_docs, word_doc_freq


if __name__ == "__main__":
    dataset = 'jigsaw'
    build_text_graph_dataset(dataset, window_size=5, use_sentiment=False, ngram_range=(1, 3))
