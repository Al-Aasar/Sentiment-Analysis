# 💬 Sentiment Analysis App

A simple **Sentiment Analysis Web App** built with **Streamlit** and a pre-trained **LSTM model**.  
The app takes a text input and predicts whether the sentiment is **Positive**, **Negative**, or **Neutral**.

---

## 🚀 Live Demo
You can try the live version of the app here:  
👉 [**View on Streamlit**](https://sentiment-analysis-project-0.streamlit.app/)  

---

## 🧠 Project Overview
This project demonstrates how to deploy a deep learning model for **Natural Language Processing (NLP)** tasks using Streamlit.  
The LSTM model was trained on text data to classify sentiment and deployed through this interactive web interface.

### 🔹 Features:
- Clean and minimal user interface  
- Real-time sentiment prediction  
- Text preprocessing (URL, mentions, punctuation removal)  
- Uses Keras LSTM model with Tokenizer and LabelEncoder  

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **TensorFlow / Keras**
- **Streamlit**
- **scikit-learn**
- **NumPy**
- **Regex**
- **Pickle**

---


## 📂 Project Structure

```
Sentiment-Analysis/
│── app.py                # Streamlit application
│── requirements.txt       # Python dependencies
│── lstm_model.keras          
│── tokenizer.pickle              
│── label_encoder.pickle          
```
