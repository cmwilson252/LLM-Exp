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
    encoded_choices = [tokenizer.encode(choice) for choice in choices_list]
    encoded_question = tokenizer.encode(question, add_special_tokens=False)
    tokenized_data.append({'input_ids': [encoded_question + choice for choice in encoded_choices]})


# Save the tokenized data to a file
save_path = "/cow02/rudenko/colowils/LLMExp/hbio.pt"
torch.save(tokenized_data, save_path)
