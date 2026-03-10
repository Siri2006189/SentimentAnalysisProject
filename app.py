import streamlit as st
import pickle
import re

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# Custom subtle styling
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4a6fa5;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    .stTextArea textarea {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💬 Sentiment Analysis ")
st.write("Analyze text sentiment using Machine Learning")

# Clean function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    return text

# Input
user_input = st.text_area("Enter your comment here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 0:
            st.error("Negative Sentiment 😞")
        elif prediction == 1:
            st.warning("Neutral Sentiment 😐")
        else:
            st.success("Positive Sentiment 😊")
    else:
        st.info("Please enter some text.")