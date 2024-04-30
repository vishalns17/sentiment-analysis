import streamlit as st
import joblib
import pandas as pd

#model = pd.read_pickle("sentiment-model.pkl")
model=joblib.load('sentiment-model.pkl')

sentiment_labels={'0':'Negative','1':'Positive'}

st.title('Sentiment Analysis')

user_input = st.text_area("Enter your text here")

if st.button('Predict'):

    predicted_sentiment = model.predict([user_input])[0]
    print("Predicted Label:"+str(predicted_sentiment))
    predicted_sentiment_label = sentiment_labels[str(predicted_sentiment)]

    st.info(f"Predicted Sentiment:{predicted_sentiment_label}")