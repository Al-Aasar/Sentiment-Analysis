import streamlit as st
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle

# تحميل النموذج وأدوات المعالجة
model = load_model('lstm_model.keras')
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
with open('labelencoder.pickle', 'rb') as f:
    labelencoder = pickle.load(f)

maxlen = 50

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower().strip()

def predict_sentiment(text):
    cleaned_text = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=maxlen, padding='post')
    pred = model.predict(padded)
    print("Model raw output:", pred)  # لتشخيص النتائج
    label = labelencoder.inverse_transform([np.argmax(pred)])
    return label[0]

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

elif option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file with a column 'text'")
    if uploaded_file:
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        if 'text' in df.columns:
            df['Sentiment'] = df['text'].apply(predict_sentiment)
            st.write(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download results as CSV", data=csv, file_name='sentiment_results.csv', mime='text/csv')
        else:
            st.warning("CSV must have a 'text' column.")
