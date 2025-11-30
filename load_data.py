from config import FLAGS
from os.path import join
from utils import get_save_path, load, save, get_corpus_path
from build_graph import build_text_graph_dataset
from dataset import TextDataset
import gc


def load_data():
    print("  → Preparing data paths...")
    dir = join(get_save_path(), 'split')
    dataset_name = FLAGS.dataset
    train_ratio = int(FLAGS.tvt_ratio[0] * 100)
    val_ratio = int(FLAGS.tvt_ratio[1] * 100)
    test_ratio = 100 - train_ratio - val_ratio
    print(f"  → Split ratio: {train_ratio}% train, {val_ratio}% val, {test_ratio}% test")
    
    if 'presplit' not in dataset_name:
        save_fn = '{}_train_{}_val_{}_test_{}_seed_{}_window_size_{}'.format(dataset_name, train_ratio,
                                                              val_ratio, test_ratio,
                                                              FLAGS.random_seed, FLAGS.word_window_size)
    else:
        save_fn = '{}_train_val_test_{}_window_size_{}'.format(dataset_name, FLAGS.random_seed, FLAGS.word_window_size)
    path = join(dir, save_fn)
    
    print(f"  → Checking for cached data: {save_fn}")
    rtn = load(path, print_msg=False)
    if rtn:
        print("  ✓ Loading from cache...")
        train_data, val_data, test_data = rtn['train_data'], rtn['val_data'], rtn['test_data']
    else:
        print("  → Cache not found, building graph from scratch...")
        train_data, val_data, test_data = _load_tvt_data_helper()
        print("  → Saving processed data to cache...")
        save({'train_data': train_data, 'val_data': val_data, 'test_data': test_data}, path, print_msg=False)
    dataset = FLAGS.dataset
    if "small" in dataset or "presplit" in dataset or 'sentiment' in dataset:
        dataset_name = "_".join(dataset.split("_")[:-1])
    else:
        dataset_name = dataset

    orig_text_path = join(get_corpus_path(), dataset_name + "_sentences.txt")
    raw_doc_list = []
    f = open(orig_text_path, 'rb')
    for line in f.readlines():
        raw_doc_list.append(line.strip().decode())
    f.close()

    return train_data, val_data, test_data, raw_doc_list


def _load_tvt_data_helper():
    print("    → Checking for full dataset cache...")
    dir = join(get_save_path(), 'all')
    path = join(dir, FLAGS.dataset + '_all_window_' + str(FLAGS.word_window_size))
    rtn = load(path, print_msg=False)
    if rtn:
        print("    ✓ Loading full dataset from cache...")
        dataset = TextDataset(None, None, None, None, None, None, rtn)
        del rtn  # Free memory from loaded cache
    else:
        print("    → Building text graph (this may take a while)...")
        dataset = build_text_graph_dataset(FLAGS.dataset, FLAGS.word_window_size)
        print("    ✓ Graph construction complete")
        gc.collect()
        print("    → Saving full dataset to cache...")
        save(dataset.__dict__, path, print_msg=False)

    print("    → Splitting into train/val/test sets...")
    train_dataset, val_dataset, test_dataset = dataset.tvt_split(FLAGS.tvt_ratio[:2], FLAGS.tvt_list, FLAGS.random_seed)
    
    # Clean up full dataset after splitting to free memory
    del dataset
    gc.collect()
    print("    ✓ Memory cleanup after split")
    
    return train_dataset, val_dataset, test_dataset