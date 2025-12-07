import scipy.sparse as sp
from math import log
from collections import Counter
from tqdm import tqdm
import gc


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
    
    # Step 2: Build document-word edges with BM25
    print("      → Building document-word edges (BM25)...")
    
    # BM25 parameters
    k1 = 1.5  # Term frequency saturation
    b = 0.75  # Length normalization
    
    # Pre-compute IDF for all words in vocabulary
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
    
    for doc in doc_list:
        tokens = [w for w in doc.split() if w in word_id_map]
        doc_lengths.append(len(tokens))
        doc_term_freqs.append(Counter(tokens))
    
    avg_doc_len = sum(doc_lengths) / len(doc_lengths)
    
    # Build edges
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

    # Step 3: Build sparse adjacency matrix
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


def build_bm25_chargram_edges_for_split(doc_list, doc_ids, total_num_docs, word_id_map, vocab, word_doc_freq):
    """
    Build graph edges for a split using BM25 + character n-grams.
    Similar to build_edges_for_split but uses BM25 instead of TF-IDF.
    """
    
    def get_char_ngrams(word, n=3):
        """Extract character n-grams from word"""
        if len(word) < n:
            return [word]
        return [word[i:i+n] for i in range(len(word) - n + 1)]
    
    # Step 1: Build word-word edges with character n-gram similarity
    print("        → Building word-word edges (character 3-grams)...")
    row = []
    col = []
    weight = []
    
    similarity_threshold = 0.4
    
    # Pre-compute character n-grams for all vocab words
    word_chargrams = {}
    for word in vocab:
        word_chargrams[word] = set(get_char_ngrams(word, 3))
    
    # Build similarity edges
    word_pair_count = 0
    for i, word_i in enumerate(vocab):
        word_i_id = word_id_map[word_i]
        ngrams_i = word_chargrams[word_i]
        
        for j in range(i + 1, len(vocab)):
            word_j = vocab[j]
            word_j_id = word_id_map[word_j]
            ngrams_j = word_chargrams[word_j]
            
            intersection = len(ngrams_i & ngrams_j)
            union = len(ngrams_i | ngrams_j)
            
            if union > 0:
                similarity = intersection / union
                
                if similarity > similarity_threshold:
                    row.append(total_num_docs + word_i_id)
                    col.append(total_num_docs + word_j_id)
                    weight.append(similarity)
                    
                    row.append(total_num_docs + word_j_id)
                    col.append(total_num_docs + word_i_id)
                    weight.append(similarity)
                    
                    word_pair_count += 1
    
    print(f"        ✓ Created {word_pair_count} word-word edge pairs")
    
    del word_chargrams
    gc.collect()
    
    # Step 2: Build document-word edges with BM25
    print("        → Building document-word edges (BM25)...")
    
    k1 = 1.5
    b = 0.75
    
    # Pre-compute IDF
    num_docs_in_split = len(doc_list)
    idf_cache = {}
    for word in vocab:
        doc_freq = word_doc_freq[word]
        idf_cache[word] = log((num_docs_in_split - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
    
    # Pre-compute document lengths
    doc_lengths = []
    doc_term_freqs = []
    
    for doc in doc_list:
        tokens = [w for w in doc.split() if w in word_id_map]
        doc_lengths.append(len(tokens))
        doc_term_freqs.append(Counter(tokens))
    
    avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1
    
    # Build edges using ORIGINAL document IDs
    for local_idx in range(len(doc_list)):
        original_doc_id = doc_ids[local_idx]
        doc_len = doc_lengths[local_idx]
        term_freq_dict = doc_term_freqs[local_idx]
        
        len_norm = k1 * (1 - b + b * doc_len / avg_doc_len)
        
        for word, term_freq in term_freq_dict.items():
            word_id = word_id_map[word]
            
            idf = idf_cache[word]
            bm25_score = idf * (term_freq * (k1 + 1)) / (term_freq + len_norm)
            
            if bm25_score > 0:
                row.append(original_doc_id)
                col.append(total_num_docs + word_id)
                weight.append(bm25_score)
    
    doc_word_edge_count = len(weight) - word_pair_count * 2
    print(f"        ✓ Created {doc_word_edge_count} document-word edges")
    
    del doc_lengths, doc_term_freqs, idf_cache
    gc.collect()

    # Build sparse matrix with FULL dimensions
    number_nodes = total_num_docs + len(vocab)
    adj_mat = sp.csr_matrix((weight, (row, col)), shape=(number_nodes, number_nodes))
    
    del row, col, weight
    gc.collect()
    
    # Make symmetric
    adj = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(adj_mat.T > adj_mat)
    
    del adj_mat
    gc.collect()
    
    return adj
