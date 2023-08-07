import pandas as pd
import torch
from transformers import AutoTokenizer

# Read the CSV file
data_csv_path = "/cow02/rudenko/colowils/LLMExp/LLM-Exp/eval/data/college_chemistry_test.csv"
data_df = pd.read_csv(data_csv_path, header=None)

# Load the tokenized data
tokenized_data = torch.load("/cow02/rudenko/colowils/LLMExp/LLM-Exp/tokenizers/tokenized_data/hchem.pt")

# Read the correct answers as text
correct_answers_text = data_df[5].tolist()

# Print the first 3 questions and their related answers + correct answer
for i in range(3):
    question_data = tokenized_data[i]
    question_text = tokenizer.decode(question_data['input_ids'][0])
    answers_text = [tokenizer.decode(choice) for choice in question_data['input_ids'][1:]]
    correct_answer_text = correct_answers_text[i]

    # Find the index of the correct answer based on the text
    correct_answer_index = answers_text.index(correct_answer_text)

    print(f"Question {i + 1}: {question_text}")
    print("Answer Choices:")
    for j, answer in enumerate(answers_text):
        print(f"{chr(97 + j)}. {answer}")
    print(f"Correct Answer: {chr(97 + correct_answer_index)}. {correct_answer_text}\n")
