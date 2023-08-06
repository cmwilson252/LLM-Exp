import pandas as pd
from transformers import AutoTokenizer
import torch

# Load the data from the spreadsheet
file_path = "/cow02/rudenko/colowils/LLMExp/LLM-Exp/eval/data/college_chemistry_test.csv"
data = pd.read_csv(file_path)
data_array = data.values
questions = data_array[:, 0].tolist()
choices = data_array[:, 1:5].tolist()
correct_answers = data_array[:, 5].tolist()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('/cow02/rudenko/colowils/LLMExp/Llama-2-7b-chat-hf')

# Tokenize the questions and choices
tokenized_data = []
for question, choices_list in zip(questions, choices):
    encoded = tokenizer(question, choices_list, return_tensors="pt", padding=False, truncation=False)
    tokenized_data.append(encoded)

# Save the tokenized data to a file
save_path = "/cow02/rudenko/colowils/LLMExp/"
torch.save(tokenized_data, save_path)
