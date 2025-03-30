import torch
import pickle
import os
from transformers import BertTokenizer, BertForSequenceClassification
import time

MODEL_PATH = "trained_model"

# Wait for the folder to be ready (max 10 sec)
timeout = 10
while not os.path.exists(MODEL_PATH) and timeout > 0:
    print(f"Waiting for {MODEL_PATH} to be available...")
    time.sleep(1)
    timeout -= 1

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Ensure app.py downloads and extracts it correctly.")

# Loading saved model, tokenizer, and label encoder
TOKENIZER_PATH = "trained_model"
LABEL_ENCODER_PATH = "trained_model/label_encoder.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer and model
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# Label encoder
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)


def predict_complaint_category(complaint_text):
    encoding = tokenizer(complaint_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    encoding = {key: val.to(device) for key, val in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    predicted_category = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_category
