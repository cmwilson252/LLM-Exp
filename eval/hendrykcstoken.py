import pandas as pd
from transformers import AutoTokenizer
import torch

# Load the data from the spreadsheet
file_path = "/path/to/your/spreadsheet.xlsx"
data = pd.read_csv(file_path)
questions = data['question_column_name'].tolist()
choices = data[['choice_a_column_name', 'choice_b_column_name', 'choice_c_column_name', 'choice_d_column_name']].values.tolist()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('llama2_model_name')

# Tokenize the questions and choices
tokenized_data = []
for question, choices_list in zip(questions, choices):
    encoded = tokenizer(question, choices_list, return_tensors="pt", padding=True, truncation=True)
    tokenized_data.append(encoded)

# Save the tokenized data to a file
save_path = "/path/to/your/tokenized_data.pt"
torch.save(tokenized_data, save_path)