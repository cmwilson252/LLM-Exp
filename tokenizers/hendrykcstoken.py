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

tokenized_data = []
for question, choice_list in zip(questions, choices):
    # Combine the question with each choice and tokenize
    tokenized_choices = [tokenizer.encode_plus(question, choice, return_tensors="pt", padding=True, truncation=True) for choice in choice_list]
    tokenized_data.append(tokenized_choices)

# Save the tokenized data to a file
save_path = "/cow02/rudenko/colowils/LLMExp/"
torch.save(tokenized_data, save_path)
