import lit as lit
import pandas as pd
import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()

def trans_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Spam Classifier")


st.markdown("<h2 style='text-align: center; color: #FF5733;'>🤖 Hey there, I'm your Spam Classifier! 🤖</h2>",
            unsafe_allow_html=True)
st.markdown(
    "<h3 style='text-align: center; color: #333333;'>Enter a message below and I'll tell you if it's Spam or Not Spam!</h3>",
    unsafe_allow_html=True)

input_sms = st.text_area("Enter the message")

if st.button('Check'):

    with st.spinner(text='checking...'):

        transformed_sms = trans_text(input_sms)

        vector_input = tfidf.transform([transformed_sms])

        result = model.predict(vector_input)[0]


    if result == 1:
        st.header("Spam")
        st.markdown("<h3 style='text-align: center; color: red;'>Watch out! It's a Spam!</h3>",
                    unsafe_allow_html=True)
    else:
        st.header("Not Spam")
        st.markdown("<h3 style='text-align: center; color: green;'>It's not a Spam! You're safe!</h3>",
                    unsafe_allow_html=True)