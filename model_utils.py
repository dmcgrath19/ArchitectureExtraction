import torch
import numpy as np
import logging
from datasets import load_dataset, get_dataset_config_names
logging.basicConfig(level='ERROR')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_pilecorpus(path, subpath_set='default', start_seed=42, split_set="train"):
    """
    Quick and ugly parsing of a WET file.
    Tested for the May 2021 crawl.
    """
    
    all_texts = ""

    available_configs = get_dataset_config_names(path)
    
    # Use 'default' if no valid subpath is provided or subpath is not in available configurations
    if subpath_set not in available_configs:
        print(f"Warning: '{subpath_set}' not found. Using 'default' instead.")
        subpath = 'default'

    print(subpath)
    print(path)
    print(start_seed)

    dataset = None
    if subpath_set is not 'default':
        dataset = load_dataset(path, subpath=subpath_set, streaming=True)
    else:
        dataset = load_dataset(path, split=split_set, streaming=True)

    shuffled_dataset = dataset.shuffle(seed=start_seed)

    dataset_head= shuffled_dataset.skip(0)
    dataset_head = shuffled_dataset.take(1000000)
    for text in dataset_head:
        all_texts+= text['text']

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