import streamlit as st
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

@st.cache_resource
def load_models():
    try:
        model = load_model('lstm_model.h5')
        
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        with open('label_encoder.pickle', 'rb') as handle:
            label_encoder = pickle.load(handle)
        
        return model, tokenizer, label_encoder
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None, None

def predict_sentiment(text, model, tokenizer, label_encoder):
    cleaned_text = clean_text(text)
    
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    
    padded = pad_sequences(sequence, maxlen=50)
    
    prediction = model.predict(padded, verbose=0)
    
    predicted_class = np.argmax(prediction, axis=1)
    confidence = np.max(prediction) * 100
    sentiment = label_encoder.inverse_transform(predicted_class)[0]
    
    return sentiment, confidence, prediction[0]

def main():

    st.title("💬 Twitter Sentiment Analysis")
    st.markdown("Enter text or a tweet to analyze the sentiment using LSTM deep learning model")
    

    with st.spinner('Loading model...'):
        model, tokenizer, label_encoder = load_models()
    
    if model is None:
        st.error("Failed to load model. Please ensure all required files are present.")
        return
    
    st.success("Model loaded successfully! ✅")
    
    st.subheader("📝 Enter Text for Analysis")
    user_input = st.text_area(
        "Write your text here:",
        height=150,
        placeholder="Example: This is an amazing product! I love it."
    )
    
    st.markdown("**Or try one of these examples:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Positive Example 😊"):
            user_input = "This is amazing! I love it so much!"
            
    with col2:
        if st.button("Negative Example 😞"):
            user_input = "This is terrible. I hate it!"
            
    with col3:
        if st.button("Neutral Example 😐"):
            user_input = "It's okay, nothing special."
    
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
    