import pandas as pd
from transformers import AutoTokenizer
import torch

# Load the data from the uploaded CSV file
file_path_csv = "/mnt/data/college_chemistry_test.csv"
data = pd.read_csv(file_path_csv)
questions = data.iloc[:, 0].tolist()
choices = data.iloc[:, 1:4].tolist()  # Adjusted to include only columns 1 to 4 for the choices

# Load the tokenizer
tokenizer_model_path = '/cow02/rudenko/colowils/LLMExp/Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)

# Tokenize the questions and choices with padding
tokenized_data_adjusted = []
for question, choices_list in zip(questions, choices):
    # Combine the question with each choice and tokenize
    combined_texts = [question + " " + str(choice) for choice in choices_list]
    encoded_data = tokenizer(combined_texts, padding='max_length', truncation=True)
    
    # Append the tokenized data for this question
    tokenized_data_adjusted.append(encoded_data)

# Save the tokenized data to a file
save_path = "/cow02/rudenko/colowils/LLMExp/LLM-Exp/tokenizers/tokenized_data/hchem.pt"
torch.save(tokenized_data_adjusted, save_path)

print("Tokenization completed and data saved.")
