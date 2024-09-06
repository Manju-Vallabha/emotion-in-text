import streamlit as st
from streamlit_lottie import st_lottie
import json
import joblib

# Load Lottie animation from a file
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Predict emotions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

# Get prediction probabilities
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Set Streamlit page configuration
st.set_page_config(page_title="Expression in Text", page_icon=":smiley:", layout="wide")

# Load animations for different emotions
happy = load_lottiefile("happy_face.json")
sadness = load_lottiefile("sad_face.json")
fear = load_lottiefile("fear_face.json")
angry = load_lottiefile("angry_face.json")
surprise = load_lottiefile("surprise.json")
neutral = load_lottiefile("neutral_face.json")
ai = load_lottiefile("ai.json")

# Load the trained emotion prediction model
pipe_lr = joblib.load(open("text_emotion.pkl", "rb"))

# Define the UI layout
title = st.container()
body = st.container()

with title:
    st.title("Emotion Recognition and Expression in Text")
    st.code("This app predicts emotions in text like happy, sadness, fear, anger, surprise, and neutral.")
    st.write("<br><br>", unsafe_allow_html=True)
with body:
    c_1, c_2, c_3 = st.columns([1, 4,1])
    with c_2:
        # Input text area for emotion prediction
        text = st.text_area("Enter Text")
        c1,c2 = st.columns([1,5])
        with c1:
            submit = st.button("Predict")

        if submit:
            # Predict emotion and display the respective animation
            prediction = predict_emotions(text)
            probability = get_prediction_proba(text)
            with c2:
                st.success(f"Predicted Emotion: {prediction}")
            
            # Display the animation based on the predicted emotion
            
            if prediction == "happy":
                    st_lottie(happy, speed=1, reverse=False, quality="low", loop=True, height=250)
            elif prediction == "sadness":
                    st_lottie(sadness, speed=1, reverse=False, quality="low", loop=True, height=250)
            elif prediction == "fear":
                    st_lottie(fear, speed=1, reverse=False, quality="low", loop=True, height=250)
            elif prediction == "anger":
                    st_lottie(angry, speed=1, reverse=False, quality="low", loop=True, height=250)
            elif prediction == "surprise":
                    st_lottie(surprise, speed=1, reverse=False, quality="low", loop=True, height=250)
            elif prediction == "neutral":
                    st_lottie(neutral, speed=1, reverse=False, quality="low", loop=True, height=250)
