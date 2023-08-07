import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Constants
PT_FILE_PATH = "/path/to/tokenized_questions.pt" # Replace with the actual path to the .pt file
CSV_PATH = "/mnt/data/college_chemistry_test.csv"
MODEL_PATH = "path_to_model" # Replace with the actual path to the Llama2 model
RESULTS_PATH = "/mnt/data/results.txt"

# Load tokenized questions from .pt file
tokenized_questions = torch.load(PT_FILE_PATH)

# Load correct answer choices from the CSV file's sixth column
csv_data = pd.read_csv(CSV_PATH)
correct_answers = csv_data.iloc[:, 5]

# Load LLM (Llama2) model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# Function to get model prediction
def get_prediction(tokenized_question):
    inputs = tokenizer(tokenized_question, return_tensors="pt")
    output = model.generate(**inputs)
    answer = tokenizer.decode(output[0])
    return answer

# Benchmarking the model
results = []
for tokenized_question, correct_answer in zip(tokenized_questions, correct_answers):
    prediction = get_prediction(tokenized_question)
    results.append((tokenized_question, prediction, correct_answer, prediction == correct_answer))

# Calculating accuracy
accuracy = sum(result[-1] for result in results) / len(results)

# Saving the results
with open(RESULTS_PATH, "w") as file:
    file.write(f"Accuracy: {accuracy}\n")
    for result in results:
        file.write(f"Question: {result[0]}\nPrediction: {result[1]}\nCorrect Answer: {result[2]}\nCorrect: {result[3]}\n\n")
