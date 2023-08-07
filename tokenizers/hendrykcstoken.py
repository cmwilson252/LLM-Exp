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

# Calculate the maximum length among all combined texts
max_length = max(len(tokenizer.tokenize(text)) for question, choices_list in zip(questions, choices)
                 for text in [question + " " + str(choice) for choice in choices_list])

# Tokenize the questions and choices with padding to the calculated maximum length
tokenized_data_adjusted = []
for question, choices_list in zip(questions, choices):
    # Tokenize each choice combined with the question
    encoded_data_choices = []
    for choice in choices_list:
        combined_text = question + " " + str(choice)
        encoded_data = tokenizer([combined_text], padding='max_length', max_length=max_length, truncation=False)
        encoded_data_choices.append(encoded_data)
    
    # Append the tokenized data for this question
    tokenized_data_adjusted.append(encoded_data_choices)

# Save the tokenized data to a file
save_path = "/cow02/rudenko/colowils/LLMExp/LLM-Exp/tokenizers/tokenized_data/hchem.pt"
torch.save(tokenized_data_adjusted, save_path)

print("Tokenization completed and data saved.")
