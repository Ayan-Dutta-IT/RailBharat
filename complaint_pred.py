import torch
import pickle
import os
import gdown
import shutil
from transformers import BertTokenizer, BertForSequenceClassification

# Google Drive Folder ID
GOOGLE_DRIVE_FOLDER_ID = "1GI9A7Xhu2bD5DsnuKv7NsAOPas-M78sx"

# Local directory for trained models
TRAINED_MODEL_DIR = "trained_model"

# Function to download trained model folder
def download_trained_model():
    if not os.path.exists(TRAINED_MODEL_DIR):
        os.makedirs(TRAINED_MODEL_DIR)

        # Download zip file from Google Drive
        zip_path = "trained_model.zip"
        gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FOLDER_ID}", zip_path, quiet=False)

        # Extract files
        shutil.unpack_archive(zip_path, TRAINED_MODEL_DIR)
        os.remove(zip_path)

download_trained_model()

# Load saved model, tokenizer, and label encoder
MODEL_PATH = "trained_model"
TOKENIZER_PATH = "trained_model"
LABEL_ENCODER_PATH = "trained_model/label_encoder.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# Load label encoder
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
