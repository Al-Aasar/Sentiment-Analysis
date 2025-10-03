import streamlit as st
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle


st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

model = load_model("lstm_model.h5")

with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pickle", "rb") as f:
    label_encoder = pickle.load(f)

max_len = 50  

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

def predict_sentiment(text, model, tokenizer, label_encoder):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")
    pred = model.predict(padded)
    label_index = np.argmax(pred)
    sentiment = label_encoder.inverse_transform([label_index])[0]
    confidence = float(np.max(pred) * 100)
    probabilities = pred[0]
    return sentiment, confidence, probabilities

st.title("💬 Sentiment Analysis App")
st.write("Enter text or a tweet to analyze the sentiment (Positive / Negative / Neutral / Irrelevant)")

user_input = st.text_area("Write your text here:",
        height=150,
        placeholder="Example: This is an amazing product! I love it.")

if st.button("🔍 Analyze Sentiment", type="primary"):
    if user_input and user_input.strip():
        with st.spinner('Analyzing...'):
            sentiment, confidence, probabilities = predict_sentiment(
                user_input, model, tokenizer, label_encoder
            )
        
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        sentiment_config = {
            'Positive': {'emoji': '😊', 'color': '#28a745'},
            'Negative': {'emoji': '😞', 'color': '#dc3545'},
            'Neutral': {'emoji': '😐', 'color': '#6c757d'},
            'Irrelevant': {'emoji': '❓', 'color': '#ffc107'}
        }
        
        config = sentiment_config.get(sentiment, {'emoji': '🤔', 'color': '#6c757d'})
        
        st.markdown(
            f"<h2 style='text-align: center; color: {config['color']};'>"
            f"{config['emoji']} {sentiment}</h2>",
            unsafe_allow_html=True
        )
        
        st.metric("Confidence Score", f"{confidence:.2f}%")
        
        st.subheader("📈 Probability Distribution")
        
        prob_data = {}
        for i, label in enumerate(label_encoder.classes_):
            prob_data[label] = probabilities[i] * 100
        
        sorted_probs = sorted(prob_data.items(), key=lambda x: x[1], reverse=True)
        
        for label, prob in sorted_probs:
            st.progress(prob / 100)
            st.write(f"**{label}**: {prob:.2f}%")
        
        with st.expander("View Cleaned Text"):
            st.text(clean_text(user_input))
            
    else:
        st.warning("⚠️ Please enter text to analyze")
