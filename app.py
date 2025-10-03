import streamlit as st
import pandas as pd
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle

# ========================
# تحميل الموديل والـ Tokenizer والـ LabelEncoder (بعد ما تحفظهم من التدريب)
# ========================
model = load_model("lstm_model.h5")

with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pickle", "rb") as f:
    label_encoder = pickle.load(f)

max_len = 50  # نفس الطول اللي استخدمته في التدريب

# ========================
# تنظيف النصوص
# ========================
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

def predict_sentiment(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")
    pred = model.predict(padded)
    label = label_encoder.inverse_transform([np.argmax(pred)])
    return label[0]

# ========================
# واجهة Streamlit
# ========================
st.title("Sentiment Analysis App 📝")

option = st.radio("اختر طريقة الإدخال:", ["اكتب نص", "ارفع ملف CSV"])

if option == "اكتب نص":
    user_input = st.text_area("اكتب الجملة هنا:")
    if st.button("تحليل"):
        if user_input.strip() != "":
            result = predict_sentiment(user_input)
            st.success(f"التصنيف: {result}")

elif option == "ارفع ملف CSV":
    file = st.file_uploader("ارفع ملف CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        if "Tweet_content" in df.columns:
            st.write("أول 5 صفوف من البيانات:")
            st.dataframe(df.head())
            if st.button("تحليل الملف"):
                df["Predicted_Sentiment"] = df["Tweet_content"].apply(predict_sentiment)
                st.success("تم تحليل البيانات ✅")
                st.dataframe(df.head(20))
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("تحميل الملف بالنتائج", data=csv, file_name="results.csv", mime="text/csv")
        else:
            st.error("لا يوجد عمود اسمه Tweet_content في الملف")
