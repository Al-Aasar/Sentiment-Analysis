import streamlit as st
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle

model = load_model('lstm_model.keras')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)



def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    return text


def predict_sentiment(text):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=50)
    prediction = model.predict(padded)
    predicted_class = np.argmax(prediction, axis=1)
    sentiment = label_encoder.inverse_transform(predicted_class)
    return sentiment[0]



st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")

st.title("💬 Sentiment Analysis App")
st.markdown("### Analyze the sentiment of any text!")

user_input = st.text_area("📝 Enter your text here:")

if st.button("🔍 Analyze"):
    if user_input.strip():
        result = predict_sentiment(user_input)

        # Define colors based on sentiment
        color_map = {
            "positive": "green",
            "negative": "red",
            "neutral": "gray",
            "irrelevant": "orange"
        }

        sentiment_lower = result.lower()
        color = color_map.get(sentiment_lower, "blue")

        st.markdown(
            f"<h3 style='text-align:center; color:{color};'>Sentiment: {result.capitalize()}</h3>",
            unsafe_allow_html=True
        )

    else:
        st.warning("⚠️ Please enter some text before analyzing.")
