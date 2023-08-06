import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer
import wandb
from transformers import AutoTokenizer

wandb.init(project="traintest")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("/cow02/rudenko/colowils/LLMExp/Llama-2-7b-chat-hf")

class TokenizedDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path, 'r') as file:
            self.data = [tokenizer.encode(line.strip()) for line in file]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])

class CustomTrainer(Trainer):
    def log(self, logs, iterator=None):
        # Log the original metrics
        super().log(logs, iterator)

        # Log the loss to wandb
        wandb.log({'loss': logs.get('loss', 0)})

        # Log the learning rate if available
        if 'learning_rate' in logs:
            wandb.log({'learning_rate': logs['learning_rate']})

# Load the dataset
train_dataset = TokenizedDataset("/cow02/rudenko/colowils/LLMExp/tokenized_data.txt", tokenizer)

# Create a DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Load the pre-trained model (modify as needed for your specific model)
model = AutoModelForMaskedLM.from_pretrained("/cow02/rudenko/colowils/LLMExp/Llama-2-7b-chat-hf")
model.to(device)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    report_to="wandb",
    learning_rate=0.01
)

# Define the Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Begin training
trainer.train()

# Save the model
trainer.save_model("./output")

wandb.config.learning_rate = training_args.learning_rate

print("Training complete. Model saved to ./output")
