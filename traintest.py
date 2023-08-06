import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer

class TokenizedDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            self.data = [list(map(int, line.strip().split())) for line in file]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])

# Load the dataset
train_dataset = TokenizedDataset('tokenized_data.txt')

# Create a DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load the pre-trained model (modify as needed for your specific model)
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Begin training
trainer.train()

# Save the model
trainer.save_model("./output")

print("Training complete. Model saved to ./output")
