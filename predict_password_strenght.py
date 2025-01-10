from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import sys
import torch

model = GPT2ForSequenceClassification.from_pretrained(sys.argv[1])

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def predict_password_strength(password):
    inputs = tokenizer(password, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    if predictions.item() == 2:
        strength="strong"
    elif predictions.item() == 1:
        strength="medium"
    else:
        strength="weak"
    return strength

# Test the model
pwd=sys.argv[2]
strength = predict_password_strength(pwd)

print(f"The password '{pwd}' is {strength}.")
