import gdown
import zipfile
import os

# Google Drive File ID
file_id = "1GI9A7Xhu2bD5DsnuKv7NsAOPas-M78sx"
zip_path = "trained_model.zip"
extract_folder = "trained_model"

# Download ZIP if not already present
if not os.path.exists(zip_path):
    print("Downloading trained model...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_path, quiet=False)

# Extract ZIP if not already extracted
if not os.path.exists(extract_folder):
    print("Extracting trained model...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)




import streamlit as st
import speech_recognition as sr
from complaint_pred import predict_complaint_category
from sentiment_pred import predict_urgency
from googletrans import Translator

# Streamlit UI Config
st.set_page_config(page_title="Railway Complaint Classification 🚆", layout="wide")

# Custom CSS for Modern UI
st.markdown(
    """
    <style>
       
        .title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            color: #0077b6;
        }

        .stTextArea textarea {
            border-radius: 10px;
            border: 2px solid #0077b6;
            padding: 10px;
            font-size: 16px;
        }

        .stButton>button {
            border-radius: 8px;
            background-color: #0077b6;
            color: white;
            padding: 10px;
            font-size: 16px;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #005f87;
        }

        .stSuccess {
            background-color: #e8f5e9;
            color: #1b5e20;
            font-weight: bold;
            padding: 15px;
            border-radius: 8px;
            font-size: 16px;
        }

        .stWarning {
            background-color: #fff3e0;
            color: #e65100;
            font-weight: bold;
            padding: 15px;
            border-radius: 8px;
            font-size: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<h1 class="title">🚆 Railway Complaint Classifier</h1>', unsafe_allow_html=True)

# Initializing session state
if "complaint_text" not in st.session_state:
    st.session_state.complaint_text = ""

# Layout
st.markdown("### 📝 Enter Your Complaint Below")
col1, col2 = st.columns([2, 1])

with col1:
    input_method = st.radio("Choose Input Method:", ["Type", "Speak"])

    if input_method == "Type":
        st.session_state.complaint_text = st.text_area("Write your complaint here:", height=150)

    elif input_method == "Speak":
        if st.button("Start Recording"):
            with sr.Microphone() as source:
                st.info("🎧 Listening... Speak now.")
                recognizer = sr.Recognizer()
                recognizer.adjust_for_ambient_noise(source)
                try:
                    audio = recognizer.listen(source, timeout=5)
                    st.session_state.complaint_text = recognizer.recognize_google(audio)
                    st.success(f"✅ Recognized Speech: {st.session_state.complaint_text}")
                except sr.UnknownValueError:
                    st.error("⚠️ Could not understand the speech.")
                except sr.RequestError:
                    st.error("❌ Google Speech API request failed.")

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/2731/2731764.png", width=120)

# Predict
st.markdown("---")
st.markdown("<h3 style='text-align: center;'>Predictions Below</h3>", unsafe_allow_html=True)
if st.button("Predict Now"):
    complaint_text = st.session_state.complaint_text

    if complaint_text:
        translator = Translator()
        detected_lang = translator.detect(complaint_text).lang

        if detected_lang != "en":
            translated_text = translator.translate(complaint_text, src=detected_lang, dest="en").text
            st.info(f"🌍 Detected Language: {detected_lang.upper()} (Translated to English)")
        else:
            translated_text = complaint_text

        # Predict category and urgency
        category = predict_complaint_category(translated_text)
        urgency = predict_urgency(translated_text)

        # Display results
        st.success(f"**Predicted Category:** {category}")
        st.warning(f"**Urgency Level:** {urgency}")
    else:
        st.error("⚠️ Please provide a complaint before predicting.")

# Footer
st.markdown("---")
st.markdown(
    "<h4 style='text-align: center; color: grey;'>🔹 Developed by The Optimizers | 🚆 RailBharat 🔹</h4>",
    unsafe_allow_html=True,
)
