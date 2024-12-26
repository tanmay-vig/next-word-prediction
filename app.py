import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app configuration
st.set_page_config(page_title="Next Word Prediction", page_icon="✍️", layout="centered", initial_sidebar_state="collapsed")

# Dark theme CSS
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stTextInput > div > input {
        background-color: #333333;
        color: #ffffff;
    }
    .stButton > button {
        background-color: #555555;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title
st.title("Next Word Prediction from Shakespeare's Hamlet ✍️")

# Input text
input_text = st.text_input("Enter the sequence of words", "To be or not to")

if st.button("Predict Next Word"):
    with st.spinner('Predicting the next word...'):
        max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.success(f'Next word: {next_word}')
