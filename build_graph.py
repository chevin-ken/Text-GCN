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


def build_text_graph_dataset(dataset, window_size):
    print(f"      → Dataset: {dataset}, Window size: {window_size}")
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

    print("      → Building graph edges (PMI & TF-IDF)...")
    sparse_graph = build_edges(doc_list, word_id_map, vocab, word_doc_freq, window_size)
    print(f"      ✓ Graph built: {sparse_graph.shape[0]} nodes, {sparse_graph.nnz} edges")
    
    # Clean up large intermediate structures
    del word_doc_freq
    gc.collect()
    print("      ✓ Memory cleanup after graph construction")
    
    docs_dict = {i: doc for i, doc in enumerate(doc_list)}
    return TextDataset(dataset, sparse_graph, labels_list, vocab, word_id_map, docs_dict, None,
                       train_test_split=split_dict)


def build_edges(doc_list, word_id_map, vocab, word_doc_freq, window_size=20):
    # constructing all windows (only with vocabulary words)
    windows = []
    for doc_words in doc_list:
        words = doc_words.split()
        # Filter to only include words in vocabulary
        words = [w for w in words if w in word_id_map]
        doc_length = len(words)
        if doc_length == 0:
            continue
        if doc_length <= window_size:
            windows.append(words)
        else:
            for i in range(doc_length - window_size + 1):
                window = words[i: i + window_size]
                windows.append(window)
    # constructing all single word frequency
    word_window_freq = defaultdict(int)
    for window in windows:
        appeared = set()
        for word in window:
            if word not in appeared:
                word_window_freq[word] += 1
                appeared.add(word)
    # constructing word pair count frequency
    word_pair_count = defaultdict(int)
    for window in tqdm(windows, desc="Building PMI"):
        for i in range(1, len(window)):
            for j in range(i):
                word_i = window[i]
                word_j = window[j]
                word_i_id = word_id_map[word_i]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_count[(word_i_id, word_j_id)] += 1
                word_pair_count[(word_j_id, word_i_id)] += 1
    
    # Store window count before deleting
    num_window = len(windows)
    
    # Clean up windows as they're no longer needed
    del windows
    gc.collect()
    
    row = []
    col = []
    weight = []

    # pmi as weights
    num_docs = len(doc_list)
    for word_id_pair, count in tqdm(word_pair_count.items()):
        i, j = word_id_pair[0], word_id_pair[1]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(num_docs + i)
        col.append(num_docs + j)
        weight.append(pmi)
    
    # Clean up word pair structures
    del word_pair_count, word_window_freq
    gc.collect()

    # frequency of document word pair (only for vocabulary words)
    doc_word_freq = defaultdict(int)
    for i, doc_words in enumerate(doc_list):
        words = doc_words.split()
        for word in words:
            if word not in word_id_map:
                continue  # Skip words not in vocabulary
            word_id = word_id_map[word]
            doc_word_str = (i, word_id)
            doc_word_freq[doc_word_str] += 1

    for i, doc_words in enumerate(doc_list):
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word not in word_id_map:
                continue  # Skip words not in vocabulary
            if word in doc_word_set:
                continue
            word_id = word_id_map[word]
            freq = doc_word_freq[(i, word_id)]
            row.append(i)
            col.append(num_docs + word_id)
            idf = log(1.0 * num_docs /
                      word_doc_freq[vocab[word_id]])
            weight.append(freq * idf)
            doc_word_set.add(word)
    
    # Clean up doc_word_freq
    del doc_word_freq
    gc.collect()

    number_nodes = num_docs + len(vocab)
    adj_mat = sp.csr_matrix((weight, (row, col)), shape=(number_nodes, number_nodes))
    
    # Clean up row, col, weight lists before final operation
    del row, col, weight
    gc.collect()
    
    adj = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(adj_mat.T > adj_mat)
    
    # Clean up intermediate matrix
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
    dataset = 'twitter_asian_prejudice'
    build_text_graph_dataset(dataset, 20)