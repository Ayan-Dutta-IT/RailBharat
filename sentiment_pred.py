from transformers import pipeline

# Loading classification for urgency detection
zero_shot_pipeline = pipeline("zero-shot-classification", model="cross-encoder/nli-deberta-v3-base")

# Urgency levels
URGENCY_LABELS = [
    "Emergency🚨",
    "High🚨",
    "Medium⚠️",
    "Low✅"
]

def predict_urgency(complaint):
    complaint_lower = complaint.lower()

    urgency_result = zero_shot_pipeline(complaint, candidate_labels=URGENCY_LABELS)
    predicted_urgency = urgency_result["labels"][0]  
    return predicted_urgency  

