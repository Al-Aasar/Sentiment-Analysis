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
with open('label_encoder.pickle', 'rb') as f:
    labelencoder = pickle.load(f)

maxlen = 50

# دالة تنظيف النص
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower().strip()

# دالة التنبؤ مع طباعة تفصيلية
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    print("Cleaned text:", cleaned_text)
    
    seq = tokenizer.texts_to_sequences([cleaned_text])
    print("Token sequence:", seq)
    
    padded = pad_sequences(seq, maxlen=maxlen, padding='post')
    print("Padded sequence shape:", padded.shape)
    print("Padded sequence:", padded)
    
    pred = model.predict(padded)
    print("Raw model prediction:", pred)
    
    predicted_label_index = np.argmax(pred)
    print("Predicted label index:", predicted_label_index)
    
    label = labelencoder.inverse_transform([predicted_label_index])
    print("Predicted label:", label[0])

    print("Label classes:", labelencoder.classes_)

    
    return label[0]

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
