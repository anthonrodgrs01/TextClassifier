import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
import io
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import gdown

# Download the tokenizer file from Google Drive
#tokenizer_url = "https://drive.google.com/file/d/1pIBILGI78uxxDLutt-I0VhuxJ-M9n9pM/view?usp=share_link"
#gdown.download(tokenizer_url, "tokenizer.json", quiet=False)

# Load the tokenizer
#with open('tokenizer.json', 'r', encoding='utf-8') as f:
#    tokenizer_json = f.read()
#tokenizer = tokenizer_from_json(tokenizer_json)

# Download the model file from Google Drive
#model_url = "https://drive.google.com/file/d/1NkU9SraEDLAmwA-mrO9NzoSMdE5GkI1z/view?usp=share_link"
#gdown.download(model_url, "CNN_MODEL.h5", quiet=False)

# Load the Keras model
#model = tf.keras.models.load_model("CNN_MODEL.h5")



# Define the Streamlit app
def app():

    # Clear cached data
    st.cache_data()

    # Clear cached resources
    st.cache_resource()
    
    # Load the tokenizer
    with open('tokenizer.json', 'r', encoding='utf-8') as f:
        tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

    # Load the model
    model = tf.keras.models.load_model('CNN_MODEL.h5')

    # Define the maximum length of input sequence
    max_len = 100

    # Define the function for predicting sentiment
    def predict_sentiment(text):
        text_sequence = tokenizer.texts_to_sequences([text])
        text_padded = pad_sequences(text_sequence, maxlen=max_len, padding='post', truncating='post')
        print(text_padded)
        prediction = model.predict(text_padded)[0]
        confidence = prediction.max() * 100
        return {'label': ['Negative', 'Neutral', 'Positive'][prediction.argmax()],
                'scores': {'Negative': prediction[0], 'Neutral': prediction[1], 'Positive': prediction[2]},
                'confidence': f'{confidence:.2f}%'}

    # Display the app title
    st.title('Text Classification using CNN')

    # Set the text input label
    st.markdown('<p style="font-size:20px;">Enter a text or upload a file to predict the sentiment:</p>', unsafe_allow_html=True)

    # Get user input
    file_type = st.selectbox("Select file type:", ("Text", "PDF", "Word"))
    if file_type == "Text":
        text = st.text_input('')
        if text:
            # Predict the sentiment
            sentiment = predict_sentiment(text)

            # Set the sentiment label icon
            if sentiment['label'] == 'Negative':
                icon = '🙁'
                color = 'red'
            elif sentiment['label'] == 'Neutral':
                icon = '😐'
                color = 'yellow'
            else:
                icon = '😃'
                color = 'green'

            # Set the sentiment label and score
            st.markdown(f'<p style="font-size:30px;">Sentiment: {sentiment["label"]} {icon}</p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size:20px;">Scores:</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:18px;color:green;">Positive: {sentiment["scores"]["Positive"]:.4f}</p>',unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:18px;color:yellow;">Neutral: {sentiment["scores"]["Neutral"]:.4f}</p>',unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:18px;color:red;">Negative: {sentiment["scores"]["Negative"]:.4f}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:20px;">Confidence: {sentiment["confidence"]}</p>', unsafe_allow_html=True)
    else:
        file = st.file_uploader("Upload file", type=["txt", "pdf", "doc", "docx"])
        if file:
            # Read the file
            if file.type == "application/pdf":
                with st.spinner('Extracting text from PDF...'):
                    pdf_reader = PyPDF2.PdfFileReader(file)
                    text = ""
                    for page in range(pdf_reader.getNumPages()):
                        text += pdf_reader.getPage(page).extractText()
            else:
                text = file.read().decode("utf-8")

                # Make sentiment prediction
                sentiment = predict_sentiment(text)

                # Set the sentiment label icon
                if sentiment['label'] == 'Negative':
                    icon = '🙁'
                    color = 'red'
                elif sentiment['label'] == 'Neutral':
                    icon = '😐'
                    color = 'yellow'
                else:
                    icon = '😃'
                    color = 'green'

                # Set the sentiment label and score
                st.markdown(f'<p style="font-size:30px;">Sentiment: {sentiment["label"]} {icon}</p>',unsafe_allow_html=True)
                st.markdown('<p style="font-size:20px;">Scores:</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-size:18px;color:green;">Positive: {sentiment["scores"]["Positive"]:.4f}</p>',unsafe_allow_html=True)
                st.markdown(f'<p style="font-size:18px;color:yellow;">Neutral: {sentiment["scores"]["Neutral"]:.4f}</p>',unsafe_allow_html=True)
                st.markdown(f'<p style="font-size:18px;color:red;">Negative: {sentiment["scores"]["Negative"]:.4f}</p>',unsafe_allow_html=True)
                st.markdown(f'<p style="font-size:20px;">Confidence: {sentiment["confidence"]}</p>',unsafe_allow_html=True)

if __name__ == "__main__":
    app()


# # Get user input
#     file_type = st.selectbox("Select file type:", ("Text", "PDF", "Word"))
#     if file_type == "Text":
#         text = st.text_input('')
#         if text:
#             # Predict the sentiment
#             sentiment = predict_sentiment(text)
#
#             # Set the sentiment label icon
#             if sentiment['label'] == 'Negative':
#                 icon = '🙁'
#                 color = 'red'
#             elif sentiment['label'] == 'Neutral':
#                 icon = '😐'
#                 color = 'yellow'
#             else:
#                 icon = '😃'
#                 color = 'green'
#
#             # Set the sentiment label and score
#             st.markdown(f'<p style="font-size:30px;">Sentiment: {sentiment["label"]} {icon}</p>', unsafe_allow_html=True)
#             st.markdown('<p style="font-size:20px;">Scores:</p>', unsafe_allow_html=True)
#             st.markdown(f'<p style="font-size:18px;color:green;">Positive: {sentiment["scores"]["Positive"]:.4f}</p>',unsafe_allow_html=True)
#             st.markdown(f'<p style="font-size:18px;color:yellow;">Neutral: {sentiment["scores"]["Neutral"]:.4f}</p>',unsafe_allow_html=True)
#             st.markdown(f'<p style="font-size:18px;color:red;">Negative: {sentiment["scores"]["Negative"]:.4f}</p>', unsafe_allow_html=True)
#             st.markdown(f'<p style="font-size:20px;">Confidence: {sentiment["confidence"]}</p>', unsafe_allow_html=True)
#
#             # Add a button to clear the input text
#             if st.button('Clear Text Input'):
#                 st.text_input('')
#
#         elif file_type == "PDF":
#             pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
#             if pdf_file is not None:
#                 pdf_reader = PyPDF2.PdfFileReader(pdf_file)
#                 text = ""
#                 for i in range(pdf_reader.getNumPages()):
#                     text += pdf_reader.getPage(i).extractText()
#                 if text:
#                     # Predict the sentiment
#                     sentiment = predict_sentiment(text)
#
#                     # Set the sentiment label icon
#                     if sentiment['label'] == 'Negative':
#                         icon = '🙁'
#                         color = 'red'
#                     elif sentiment['label'] == 'Neutral':
#                         icon = '😐'
#                         color = 'yellow'
#                     else:
#                         icon = '😃'
#                         color = 'green'
#
#                     # Set the sentiment label and score
#                     st.markdown(f'<p style="font-size:30px;">Sentiment: {sentiment["label"]} {icon}</p>',
#                                 unsafe_allow_html=True)
#                     st.markdown('<p style="font-size:20px;">Scores:</p>', unsafe_allow_html=True)
#                     st.markdown(
#                         f'<p style="font-size:18px;color:green;">Positive: {sentiment["scores"]["Positive"]:.4f}</p>',
#                         unsafe_allow_html=True)
#                     st.markdown(
#                         f'<p style="font-size:18px;color:yellow;">Neutral: {sentiment["scores"]["Neutral"]:.4f}</p>',
#                         unsafe_allow_html=True)
#                     st.markdown(
#                         f'<p style="font-size:18px;color:red;">Negative: {sentiment["scores"]["Negative"]:.4f}</p>',
#                         unsafe_allow_html=True)
#                     st.markdown(f'<p style="font-size:20px;">Confidence: {sentiment["confidence"]}</p>',
#                                 unsafe_allow_html=True)
#
#                     # Add a button to clear the input file
#                     if st.button('Clear File Input'):
#                         pdf_file = None
#
#         else:
#             st.write('File type not supported')


# elif file_type == "Word":
# word_file = st.file_uploader("Upload a Word file", type=["docx", "doc"])
# if word_file is not None:
# import docx2txt
# text = docx2txt.process(word_file)
# if text:
# # Predict the sentiment
# sentiment = predict_sentiment(text)