import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import requests
import json

# Define the example text
example = "Please, help me."

# Load the RoBERTa model and tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# Function to calculate polarity scores using the RoBERTa model
def polarity_scores_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)  # Apply softmax to get probabilities
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict


# Function to map sentiment scores to emotions
def map_to_emotion(positive, negative, neutral):
    if positive > 0.6:
        if positive > 0.8:
            return "Exclamation"
        return "Happiness"
    elif negative > 0.6:
        if neutral > 0.4:
            return "Sadness"
        elif negative > 0.8:
            return "Distress"
        else:
            return "Anger"
    elif neutral > 0.6:
        return "Neutral"
    else:
        return "Mixed Emotion"


# Function to get sentiment scores and map them to emotions
def get_emotion_from_text(text):
    scores = polarity_scores_roberta(text)
    emotion = map_to_emotion(scores['roberta_pos'], scores['roberta_neg'], scores['roberta_neu'])
    print(f"Text: {text}")
    print(f"Scores: {scores}")
    print(f"Emotion: {emotion}")
    return scores, emotion


# Function to generate TTS with ElevenLabs
def generate_tts(text, emotion):
    api_key = "sk_990064973042ebefcf7d6a37b01c087a80ac79b4f67ba90c"
    voice_id = "29vD33N1CtxCmqQRPOHJ"

    # Define voice settings based on emotion
    if emotion == "Happiness":
        stability = 0.8
        similarity_boost = 0.9
        style = 0.7
    elif emotion == "Sadness":
        stability = 0.5
        similarity_boost = 0.9
        style = 0.3
    elif emotion == "Anger":
        stability = 0.7
        similarity_boost = 0.9
        style = 0.6
    elif emotion == "Exclamation":
        stability = 0.6
        similarity_boost = 0.9
        style = 0.9
    elif emotion == "Distress":
        stability = 0.4
        similarity_boost = 0.8
        style = 0.3
    elif emotion == "Neutral":
        stability = 0.5
        similarity_boost = 0.5
        style = 0.5
    else:  # Mixed Emotion
        stability = 0.5
        similarity_boost = 0.5
        style = 0.5

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": True
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        with open("output_audio.mp3", "wb") as audio_file:
            audio_file.write(response.content)
        print("Audio generated and saved as output_audio.mp3")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


# Test the function with the example text
example_scores, example_emotion = get_emotion_from_text(example)
generate_tts(example, example_emotion)

# Test with a new text
new_text = "I am in great danger."
new_text_scores, new_text_emotion = get_emotion_from_text(new_text)
generate_tts(new_text, new_text_emotion)


