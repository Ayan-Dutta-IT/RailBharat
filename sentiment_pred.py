from transformers import pipeline

# Load zero-shot classification for urgency detection
zero_shot_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define urgency levels
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

