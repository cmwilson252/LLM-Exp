import torch
import pandas as pd
from transformers import AutoTokenizer, LlamaPreTrainedModel

# Constants
PT_FILE_PATH = "/cow02/rudenko/colowils/LLMExp/LLM-Exp/tokenizers/tokenized_data/hchem.pt" # Replace with the actual path to the .pt file
CSV_PATH = "/cow02/rudenko/colowils/LLMExp/LLM-Exp/eval/data/college_chemistry_test.csv"
MODEL_PATH = "/cow02/rudenko/colowils/LLMExp/Llama-2-7b-chat-hf" # Replace with the actual path to the Llama2 model
RESULTS_PATH = "/cow02/rudenko/colowils/LLMExp/LLM-Exp/results.txt"

# Load tokenized questions from .pt file
tokenized_questions = torch.load(PT_FILE_PATH)

# Load correct answer choices from the CSV file's sixth column
csv_data = pd.read_csv(CSV_PATH)
correct_answers = csv_data.iloc[:, 5]

# Load LLM (Llama2) model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = LlamaPreTrainedModel.from_pretrained(MODEL_PATH)

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
