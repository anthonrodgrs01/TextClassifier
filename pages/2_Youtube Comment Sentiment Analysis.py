import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
from googleapiclient.discovery import build
from langdetect import detect, LangDetectException
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import gdown
from streamlit_lottie import st_lottie
import requests

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
    prediction = model.predict(text_padded)[0]
    confidence = prediction.max() * 100
    return {'label': ['Negative☹️', 'Neutral😐', 'Positive😀'][prediction.argmax()],
            'scores': {'Negative': prediction[0], 'Neutral': prediction[1], 'Positive': prediction[2]},
            'confidence': f'{confidence:.2f}%'}


# Define the function for analyzing the comments of a YouTube video
def analyze_youtube_comments(channel_name, video_url, analyze_comments):
    # Set up the YouTube API client
    api_key = 'AIzaSyAIXyIZ65HCpF8ke6LUdwDmTgiSvkcQOMc'  # Enter your YouTube API key here
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Get the channel ID from the user
    search_response = youtube.search().list(
        q=channel_name,
        type='channel',
        part='id,snippet'
    ).execute()
    channel_id = search_response['items'][0]['id']['channelId']

    # Get the video ID from the URL
    video_id = video_url.split('=')[-1]

    # Fetch the comments for the video
    comments = []
    next_page_token = None
    while len(comments) < analyze_comments:
        try:
            comment_response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                maxResults=min(100, analyze_comments - len(comments)),
                pageToken=next_page_token
            ).execute()
        except:
            break
        for item in comment_response['items']:
            try:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                language = detect(comment)
                if language == 'en':
                    comments.append(comment)
            except LangDetectException:
                pass
        if 'nextPageToken' in comment_response:
            next_page_token = comment_response['nextPageToken']
        else:
            break

    # Analyze the comments for sentiment
    results = []
    for comment in comments:
        result = predict_sentiment(comment)
        result['Comment'] = comment
        results.append(result)
    return results


# Define the Streamlit app
def app():
    st.balloons()

    # Function to load Lottie animation from URL
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    # Load and display the Lottie animation in the first column
    col1, col2 = st.columns([1, 3])
    with col1:
        st_lottie(load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_ej2lfhv2.json"), speed=1, width=175,height=175, key="hello")

    # Set the page title in the second column
    with col2:
        st.title('YouTube Comment Sentiment Analysis')

    # Get the channel name and video URL from the user
    channel_name = st.text_input('Enter the name of the YouTube channel:')
    video_url = st.text_input('Enter the URL of the YouTube video:')
    analyze_comments = st.slider('Select the number of comments to analyze:', 10, 50, 25)

    # Analyze the comments and display the results
    # Analyze the comments and display the results
    if st.button('Analyze Comments'):
        try:
            results = analyze_youtube_comments(channel_name, video_url, analyze_comments)
            df = pd.DataFrame(results)
            df = df.rename(columns={'label': 'Sentiment Result', 'confidence': 'Confidence'})
            df.index += 1
            df = df[['Comment', 'Sentiment Result', 'Confidence']]
            st.write(df)
        except Exception as e:
            st.error(f'Error: {e}')


# Run the Streamlit app

if __name__ == '__main__':
    app()