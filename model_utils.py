import torch
import numpy as np
import logging
from datasets import load_dataset, get_dataset_config_names
logging.basicConfig(level='ERROR')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_pilecorpus(path,start_seed=42):
    """
    This is a way for parsing the Pile corpus.
    """
    
    all_texts = ""
    dataset = load_dataset(path, split="train", streaming=True)
    shuffled_dataset = dataset.shuffle(seed=start_seed)
    dataset_head= shuffled_dataset.skip(0)
    dataset_head = shuffled_dataset.take(1000000)

    for text in dataset_head:
        all_texts+= text['text']

    return all_texts

def parse_splitted(path, subset='default', start_seed=42):
    """
    This is for parsing thePileSplitted dataset.
    """
    
    all_texts = ""

    print(f"Subset: {subset}")
    print(f"Path: {path}")
    print(f"Start Seed: {start_seed}")

    # Load dataset with streaming enabled
    dataset = load_dataset(path, subset, streaming=True)

    for idx, example in enumerate(dataset):
        if idx >= 1000000:  # Limiting to 1,000,000 examples
            break
        all_texts += example['text']

    return all_texts


def parse_wmt_splitted(path, split_set='train'):
    """
    This is for getting data from KaiNylund/WMT-year-splits
    """
    all_texts = ""
    
    # Load the dataset split with streaming enabled
    dataset = load_dataset(path, split=split_set, streaming=True)
    
    # Iterate over the dataset split and accumulate texts
    for idx, example in enumerate(dataset):
        all_texts += example['text']
        
        # Optional: Limit the number of examples processed
        if idx >= 1000000:
            break
    
    return all_texts


def calculate_perplexity(sentence, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss)

def print_best(metric, samples, name1, scores1, name2=None, scores2=None, n=10):
    """
    Print the `n` best samples according to the given `metric`.
    Returns a string containing the information for each sample.
    """
    idxs = np.argsort(metric)[::-1][:n]
    output_string = ""

    for i, idx in enumerate(idxs):
        if scores2 is not None:
            sample_info = f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}"
        else:
            sample_info = f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}"

        sample_text = samples[idx]
        output_string += sample_info + "\n" + sample_text + "\n\n"

    return output_string