import argparse
import numpy as np
import sys
import torch
import zlib
import csv
from tqdm import tqdm
from datasets import load_dataset
from model_utils import load_model, generate_samples, calculate_perplexity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_pilecorpus():
    """
    Load the Pile dataset and return the concatenated text.
    """
    dataset = load_dataset('the_pile', split='train')
    all_texts = " ".join(dataset['text'][:10])  # Adjust this range as needed
    return all_texts

def save_to_csv(samples, scores, filename):
    """
    Save the samples and scores to a CSV file.
    """
    fields = ['Sample', 'PPL']
    rows = zip(samples, scores)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)

def main(args):
    print(f"using device: {device}")

    if args.custom_sampling:
        print("Loading Pile dataset...")
        cc = parse_pilecorpus()
        print("Length:", len(cc))

    seq_len = 256
    top_k = 40

    model, tokenizer = load_model(args.model, args.model_type)

    samples = []
    scores = []

    num_batches = int(np.ceil(args.N / args.batch_size))
    with tqdm(total=args.N) as pbar:
        for i in range(num_batches):
            if args.custom_sampling:
                input_len = 10
                input_ids = []
                attention_mask = []

                while len(input_ids) < args.batch_size:
                    r = np.random.randint(0, len(cc))
                    prompt = " ".join(cc[r:r+100].split(" ")[1:-1])
                    inputs = tokenizer(prompt, return_tensors="pt", max_length=input_len, truncation=True)
                    if len(inputs['input_ids'][0]) == input_len:
                        input_ids.append(inputs['input_ids'][0])
                        attention_mask.append(inputs['attention_mask'][0])

                inputs = {'input_ids': torch.stack(input_ids), 'attention_mask': torch.stack(attention_mask)}
                prompts = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            else:
                prompts = [""] * args.batch_size

            texts = generate_samples(model, tokenizer, prompts, seq_len, top_k)

            for text in texts:
                p = calculate_perplexity(text, model, tokenizer)
                zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

                samples.append(text)
                scores.append(p)

            pbar.update(args.batch_size)

    scores = np.asarray(scores)

    save_to_csv(samples, scores, args.output_file)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000, help="Number of samples to generate")
    parser.add_argument('--batch-size', type=int, default=10, help="Batch size for generation")
    parser.add_argument('--custom-sampling', action='store_true', help="Condition the generation using custom dataset")
    parser.add_argument('--model', type=str, required=True, help="Model to use (e.g., 'EleutherAI/pythia-410m')")
    parser.add_argument('--model_type', type=str, choices=['pythia', 'gptneo'], default='gptneo', help="Type of the model")
    parser.add_argument('--output-file', type=str, default='samples.csv', help="Output CSV file to save results")
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)


# python main.py --N 1000 --batch-size 10 --custom-sampling --model EleutherAI/pythia-410m --model_type pythia --output-file results-pythia410.csv