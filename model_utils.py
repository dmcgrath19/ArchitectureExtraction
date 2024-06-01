# model_utils.py
import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, GPTNeoXForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_name, model_type='gptneo'):
    """
    Load the specified model and tokenizer from Hugging Face.
    """
    if model_type == 'pythia':
        model_cls = GPTNeoXForCausalLM
    else:
        model_cls = GPTNeoForCausalLM

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = model_cls.from_pretrained(model_name, return_dict=True).to(device)
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()
    
    return model, tokenizer

def generate_samples(model, tokenizer, prompts, seq_len=256, top_k=40):
    """
    Generate text samples using the specified model and tokenizer.
    """
    input_len = 1
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    output_sequences = model.generate(
        input_ids=inputs['input_ids'].to(device),
        attention_mask=inputs['attention_mask'].to(device),
        max_length=input_len + seq_len,
        do_sample=True, 
        top_k=top_k, 
        top_p=1.0
    )
    texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    return texts

def calculate_perplexity(sentence, model, tokenizer):
    """
    Calculate the perplexity of a sentence using the given model and tokenizer.
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    return torch.exp(loss)
