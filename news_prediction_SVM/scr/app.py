import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
import joblib

# Function to preprocess text
def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])

# Load your trained model
# Replace 'email_prediction_model.pkl' with the actual filename of your trained model
# You need to have joblib installed: pip install joblib
loaded_model = joblib.load('news_prediction_model.pkl')

def main():
    st.title("News Prediction Classification")

    # User input for email
    user_news = st.text_area("Enter an news:", "")

    if st.button("Predict"):
        if user_news:
            # Preprocess the input email
            processed_news = text_process(user_news)
            st.write("Processed Email:", processed_news)

            # Tokenize and transform using CountVectorizer
            bow = loaded_model.named_steps['bow'].transform([processed_news])
            st.write("After CountVectorizer:", bow)

            # Transform using TfidfTransformer
            tfidf = loaded_model.named_steps['tfidf'].transform(bow)
            st.write("After TfidfTransformer:", tfidf)

            # Make a prediction and get probability scores using the classifier
            classifier = loaded_model.named_steps['classifier']
            prediction_value = classifier.predict(tfidf)[0]
            probability_spam = classifier.predict_proba(tfidf)[0, 1]
            st.write("Prediction:", prediction_value)


if __name__ == "__main__":
    main()
