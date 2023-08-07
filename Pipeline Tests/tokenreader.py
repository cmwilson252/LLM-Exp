import pandas as pd
import torch
from transformers import AutoTokenizer

# Read the CSV file
data_csv_path = "/cow02/rudenko/colowils/LLMExp/LLM-Exp/eval/data/college_chemistry_test.csv"
data_df = pd.read_csv(data_csv_path, header=None)

# Load the tokenized data
tokenized_data = torch.load("/cow02/rudenko/colowils/LLMExp/LLM-Exp/tokenizers/tokenized_data/hchem.pt")


# Path to your downloaded tokenizer
tokenizer_path = "/cow02/rudenko/colowils/LLMExp/Llama-2-7b-chat-hf"
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Read the correct answers as text
correct_answer_letters = data_df[5].tolist()

# Print the first 3 questions and their related answers + correct answer
for i in range(3):
    question_data = tokenized_data[i]
    question_text = tokenizer.decode(question_data['input_ids'][0])
    answers_text = [tokenizer.decode(choice) for choice in question_data['input_ids'][1:]]
    correct_answer_letter = correct_answer_letters[i]
    
    # Convert the correct answer letter to an index (0 for "A", 1 for "B", etc.)
    correct_answer_index = ord(correct_answer_letter) - ord('A')

    print(f"Question {i + 1}: {question_text}")
    print("Answer Choices:")
    for j, answer in enumerate(answers_text):
        print(f"{chr(97 + j)}. {answer}")
    print(f"Correct Answer: {chr(97 + correct_answer_index)}. {answers_text[correct_answer_index]}\n")
