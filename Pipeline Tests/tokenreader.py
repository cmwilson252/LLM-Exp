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

# Print the first 3 questions
for i in range(3):
    question_data = tokenized_data[i]
    question_text = tokenizer.decode(question_data['input_ids'][0])
    answers_text = [tokenizer.decode(choice) for choice in question_data['input_ids'][1:5]]  # Limit to 4 answer choices
    correct_answer_letter = correct_answer_letters[i]
    correct_answer_index = ord(correct_answer_letter) - ord('A')

    # Check if the index is within the range of the answers
    if 0 <= correct_answer_index < len(answers_text):
        print(f"Question {i + 1}: {question_text}")
        print("Answer Choices:")
        for j, answer in enumerate(answers_text):
            print(f"{chr(97 + j)}. {answer}")
        print(f"Correct Answer: {chr(97 + correct_answer_index)}. {answers_text[correct_answer_index]}\n")
    else:
        print(f"Error with question {i + 1}: Correct answer index is out of range.\n")
