from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys
import torch
import torch.nn as nn
import pandas as pd

# load the dataset
# load weak password dataset
f=open(sys.argv[1],"r")

lines=f.readlines()
weak_password_list = [line.strip() for line in lines]
weak_password_label=[0]*len(weak_password_list)

f.close()
# load strong password dataset
f=open(sys.argv[2],"r")

lines=f.readlines()
medium_password_list = [line.strip() for line in lines]
medium_password_label=[1]*len(medium_password_list)

f.close()

f=open(sys.argv[3],"r")

lines=f.readlines()
strong_password_list = [line.strip() for line in lines]
strong_password_label=[2]*len(strong_password_list)

f.close()

passwords= weak_password_list + medium_password_list + strong_password_list
labels= weak_password_label + medium_password_label + strong_password_label

data = {
    "password": passwords,
    "label": labels
}

# Load the tokenizer and model 
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)
model.resize_token_embeddings(len(tokenizer))

# Explicitly set pad_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# Convert data to tensors
passwords = data["password"]
labels = torch.tensor(data["label"])

encodings = tokenizer(passwords, padding=True, truncation=True, return_tensors="pt")

class PasswordDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings  # Tokenized password data
        self.labels = labels        # Corresponding labels (0 for weak, 1 for strong)

    def __getitem__(self, idx):
        # Retrieve the tokenized password and label at the specified index
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.labels)

class CustomWeightedLoss(nn.Module):
    def __init__(self,weights):
        super(CustomWeightedLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weights)

    def forward(self, outputs, targets):
        """ outputs: Tensor of shape (batch_size, num_classes) targets: Tensor of shape (batch_size) """
        loss = self.criterion(outputs, targets)
        return loss

class CustomTrainer(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, compute_metrics, weights):
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics)
        self.loss_fn = CustomWeightedLoss(weights)
    
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

train_dataset = PasswordDataset(encodings, labels)

class_weights = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch"
)

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    accuracy = accuracy_score(p.label_ids, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds)
    return {"accuracy": accuracy, "precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist()}


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    compute_metrics=compute_metrics,
    weights=class_weights )


trainer.train()

