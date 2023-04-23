import re
import nltk
import pickle
import streamlit as st
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

model = pickle.load(open('model/model.pkl', 'rb'))
cv = pickle.load(open('model/cv.pkl', 'rb'))

def tokenize(email):
    email = email.lower()
    email = re.sub(r'[^a-zA-Z0-9]', ' ', email)
    email = email.split()
    lemmatizer = WordNetLemmatizer()
    email = [lemmatizer.lemmatize(word) for word in email if word not in stopwords.words('english')]
    email = ' '.join(email)
    return email

def predict(email):
    email = tokenize(email)
    email = cv.transform([email]).toarray()
    prediction = model.predict(email)
    return prediction[0]

def main():
    st.title(':sparkles: E-Mail Spam Classifier :sparkles:')
    st.subheader('Enter the E-Mail below to check if it is Spam or Ham')
    email = st.text_input('Email')
    if st.button('Predict'):
        result = predict(email)
        if result == 0:
            st.success('Ham :white_check_mark:')
        else:
            st.error('Spam :heavy_exclamation_mark:')

if __name__ == '__main__':
    main()