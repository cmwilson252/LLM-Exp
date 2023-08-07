import pandas as pd
import torch
from transformers import AutoTokenizer

# Read the CSV file
data_csv_path = "/cow02/rudenko/colowils/LLMExp/LLM-Exp/eval/data/college_chemistry_test.csv"
data_df = pd.read_csv(data_csv_path, header=None)

# Extract the correct answers as column indices (2, 3, 4, 5)
correct_answers = data_df[5].tolist()

# Load the tokenized data
tokenized_data = torch.load("/cow02/rudenko/colowils/LLMExp/LLM-Exp/tokenizers/tokenized_data/hchem.pt")

# Print the first 3 questions and their related answers + correct answer
for i in range(3):
    question_data = tokenized_data[i]
    question_text = tokenizer.decode(question_data['input_ids'][0])
    answers_text = [tokenizer.decode(choice) for choice in question_data['input_ids'][1:]]
    correct_answer_index = correct_answers[i] - 2  # Adjusting the index to 0-based
    correct_answer_text = answers_text[correct_answer_index]

    print(f"Question {i + 1}: {question_text}")
    print("Answer Choices:")
    for j, answer in enumerate(answers_text):
        print(f"{chr(97 + j)}. {answer}")
    print(f"Correct Answer: {chr(97 + correct_answer_index)}. {correct_answer_text}\n")
