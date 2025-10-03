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

# دالة تنظيف النص
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)  
    text = re.sub(r"[^a-zA-Z\s]", "", text) 
    text = text.lower().strip()
    return text

# دالة التنبؤ مع طباعة تفصيلية
def predict_sentiment(text):
    cleaned_text = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=50)

    prediction = model.predict(padded)
    predicted_class = np.argmax(prediction, axis=1)
    sentiment = label_encoder.inverse_transform(predicted_class)
    
    return sentiment[0]

# واجهة المستخدم
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
