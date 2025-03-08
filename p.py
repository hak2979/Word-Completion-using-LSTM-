import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def predict_next_words(model, tokenizer, input_text, num_words=5, max_length=5):
    # Preprocess input_text to match the training format
    input_seq = tokenizer.texts_to_sequences([input_text])[0]
    
    # Prepare a list to hold the predicted words
    predicted_words = []

    for _ in range(num_words):
        # Pad the input sequence
        input_seq = pad_sequences([input_seq], maxlen=max_length, padding='pre')
        
        # Predict the next word
        predicted = model.predict(input_seq, verbose=0)
        
        # Get the index of the predicted word
        predicted_word_index = np.argmax(predicted, axis=-1)[0]
        
        # Check if the predicted index is in the tokenizer index
        if predicted_word_index in tokenizer.index_word:
            # Convert the index to the actual word
            predicted_word = tokenizer.index_word[predicted_word_index]
        else:
            predicted_word = "<unknown>"  # Handle unknown word

        # Append the predicted word to the list
        predicted_words.append(predicted_word)
        
        # Update the input sequence with the predicted word
        input_seq = np.append(input_seq[0][1:], predicted_word_index)  # Remove the first word and append the new word
    
    return ' '.join(predicted_words)

# Streamlit UI setup
st.title("Auto Word Predictor")
st.write("Enter a partial sentence below to get word predictions:")

# Load the model and tokenizer
model_ = load_model("lstm_next_word_model.keras")
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# User inputs
input_text = st.text_input("Write your text here:")
num_words = st.text_input("Number of words to predict (default is 5):", value="5")

# Prediction button
if st.button("Predict"):
    if input_text and num_words:
        try:
            # Convert num_words to an integer
            num_words = int(num_words)
            # Predict next words
            ans = predict_next_words(model_, tokenizer, input_text, num_words)
            st.write(f"Predicted words: {ans}")
        except ValueError:
            st.write("Please enter a valid number for the number of words to predict.")
        except Exception as e:
            st.write(f"An error occurred: {str(e)}")