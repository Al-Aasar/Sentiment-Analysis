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

maxlen = 50


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


st.title("Sentiment Analysis App")

option = st.radio("Choose input method:", ("User Input", "Upload CSV"))

if option == "User Input":
    user_input = st.text_area("Enter your text here:")
    if st.button("Analyze"):
        if user_input.strip():
            result = predict_sentiment(user_input)
            st.success(f"Sentiment: {result}")
        else:
            st.warning("Please enter some text.")
