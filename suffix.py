import torch
import csv
import pandas as pd
import zlib
from datetime import datetime
from transformers import AutoTokenizer, RwkvForCausalLM, StoppingCriteria
import argparse


class RwkvStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence=[187, 187], eos_token_id=537):
        self.eos_sequence = eos_sequence
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_2_ids = input_ids[:, -2:].tolist()
        return self.eos_sequence in last_2_ids


def clean_prompt(prompt):
    # Remove the first and last double quotes if they exist
    if prompt.startswith('"') and prompt.endswith('"'):
        prompt = prompt[1:-1]
    
    # Replace doubled quotes with single quotes
    prompt = prompt.replace('""', '"')
    
    return prompt


def main(model_name, input_csv_path):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = RwkvForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Define parameters for generation
    seq_len = 50  # Number of tokens to generate

    now = datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M")  # Format: YYYYMMDD_HHMM

    model_name_short = model_name.split('/')[-1]
    output_csv_path = f'rwkv-{model_name_short}_{time_str}.csv'

    # Read the input CSV
    df = pd.read_csv(input_csv_path)

    # Create a list to store the rows for the new CSV
    output_rows = []

    # Loop through each row in the dataframe
    for index, row in df.iterrows():
        prompt = clean_prompt(row['prompt'])  # Assuming your CSV has a column named 'prompt'
        sample = clean_prompt(row['sample'])  # Assuming your CSV has a column named 'sample'
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(device)
        
        # Generate text
        output_sequences = model.generate(
            input_ids=inputs['input_ids'],
            max_new_tokens=seq_len,
            stopping_criteria=[RwkvStoppingCriteria()]
        )
        
        # Decode the generated text
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        print(generated_text + '\n\n')

        zscore = len(zlib.compress(generated_text.encode('utf-8')))
        # ppl = calculate_perplexity(generated_text, model, tokenizer)

        # Append the sample, prompt, and generated text to the list
        output_rows.append({
            'sample': sample,
            'prompt': prompt,
            'suffix': generated_text,
            'Zlib': zscore
        })

    # Save the output to a new CSV
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_MINIMAL, quotechar='"', escapechar='\\')

    print(f"Output saved to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using RWKV model.")
    parser.add_argument(
        '--model_name', 
        type=str, 
        required=True, 
        help="The model name or path to the RWKV model"
    )
    parser.add_argument(
        '--input_csv', 
        type=str, 
        required=True, 
        help="Path to the input CSV file"
    )
    
    args = parser.parse_args()
    
    main(args.model_name, args.input_csv)
