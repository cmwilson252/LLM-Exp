import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, TrainingArguments, Trainer
import wandb
from transformers import AutoTokenizer

wandb.init(project="traintest")

class MinimalDataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.texts = ["A"]  # Just one token
        self.labels = [0]   # Corresponding label

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=10)
        encoding['labels'] = torch.tensor(label, dtype=torch.long)
        return encoding

tokenizer = AutoTokenizer.from_pretrained('llama path')
model = AutoModel.from_pretrained('llama path')

train_dataset = MinimalDataset(tokenizer)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

training_args = TrainingArguments(
    output_dir='./output',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    logging_dir='./logs',
    logging_steps=1,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

print("Training complete. Model saved to ./output")
