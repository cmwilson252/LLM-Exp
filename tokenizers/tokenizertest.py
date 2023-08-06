from datasets import load_dataset
from transformers import AutoTokenizer
import os

# Function to extract and join text
def extract_and_join_text(example):
    title = example.get('title', '')
    abstract = example.get('abstract', '')
    body_text_sentences = example.get('body_text', [])
    body_text = ' '.join(body_text_sentences)
    full_text = f"{title} {abstract} {body_text}"
    return full_text

# Load the dataset
dataset = load_dataset("orieg/elsevier-oa-cc-by")

# Path to your downloaded tokenizer
tokenizer_path = "/cow02/rudenko/colowils/LLMExp/Llama-2-7b-chat-hf"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Tokenize the dataset and save to a file
with open('tokenized_data.txt', 'w') as file:
    for example in dataset['train']:
        text = extract_and_join_text(example)
        tokens = tokenizer.tokenize(text)
        file.write(' '.join(tokens) + '\n')
        print("Tokenized an example...") # Print confirmation

print("Tokenization complete. Data saved to tokenized_data.txt")
