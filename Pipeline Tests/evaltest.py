import pandas as pd
import torch
from transformers import LlamaModel
from torch.utils.data import DataLoader

# Load the tokenized data
tokenized_data = torch.load("/cow02/rudenko/colowils/LLMExp/LLM-Exp/tokenizers/tokenized_data/hbio.pt")

# Load the Excel file without headers to extract the correct answers
file_path = "/cow02/rudenko/colowils/LLMExp/LLM-Exp/eval/data/college_chemistry_test.csv"
data = pd.read_csv(file_path, header=None)
data_array = data.values
# Map the letter choices to numerical indices
correct_answers_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
correct_answers = [correct_answers_map[answer.lower()] for answer in data_array[:, 5]]

# Load the model
model = LlamaModel.from_pretrained('/cow02/rudenko/colowils/LLMExp/Llama-2-7b-chat-hf')
model.eval()  # Set the model to evaluation mode

# Define a custom dataset class
class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data, correct_answers):
        self.tokenized_data = tokenized_data
        self.correct_answers = correct_answers
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        item = {key: value.squeeze() for key, value in self.tokenized_data[idx].items()}
        label = self.correct_answers[idx]
        return item, label

# Create the dataset and DataLoader with padding
from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    inputs = {key: pad_sequence([item[key] for item in batch[0]], batch_first=True) for key in batch[0][0].keys()}
    labels = torch.tensor(batch[1])
    return inputs, labels

# Create the dataset and DataLoader
dataset = TokenizedDataset(tokenized_data, correct_answers)
dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)  # Adjust batch size as needed

# Evaluate the model
correct = 0  # Count of correct predictions
total = 0   # Total number of questions

with torch.no_grad():
    for batch, labels in dataloader:
        # Prepare the inputs for the model
        inputs = {key: torch.stack([item[key] for item in batch], dim=0) for key in batch[0].keys()}
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Determine the predicted answers
        predictions = torch.argmax(logits, dim=1)
        
        # Calculate accuracy
        correct += (predictions == labels).sum().item()
        total += len(labels)

accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")
