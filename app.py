import os
from flask import Flask, request, render_template, jsonify, send_file
import json
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from pymongo import MongoClient

app = Flask(__name__)

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client.text_to_speech
audio_collection = db.audios

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

API_KEY = "sk_990064973042ebefcf7d6a37b01c087a80ac79b4f67ba90c"

def polarity_scores_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

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

def generate_tts(text, emotion, voice_id="29vD33N1CtxCmqQRPOHJ"):
    voice_settings = {
        "Happiness": {"stability": 0.3, "similarity_boost": 0.9, "style": 0.9},
        "Sadness": {"stability": 0.7, "similarity_boost": 0.8, "style": 0.2},
        "Anger": {"stability": 0.4, "similarity_boost": 0.8, "style": 0.7},
        "Exclamation": {"stability": 0.3, "similarity_boost": 0.9, "style": 1.0},
        "Distress": {"stability": 0.8, "similarity_boost": 0.7, "style": 0.3},
        "Neutral": {"stability": 0.6, "similarity_boost": 0.6, "style": 0.5},
        "Mixed Emotion": {"stability": 0.5, "similarity_boost": 0.7, "style": 0.6}
    }.get(emotion, {"stability": 0.6, "similarity_boost": 0.6, "style": 0.5})

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {"Content-Type": "application/json", "xi-api-key": API_KEY}
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {**voice_settings, "use_speaker_boost": True}
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        audio_path = f"static/{text[:10]}_audio.mp3"
        with open(audio_path, "wb") as audio_file:
            audio_file.write(response.content)
        return audio_path
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        text = data['text']
        emotion = data['emotion']
        voice_id = data['voice_id']
        
        scores = polarity_scores_roberta(text)
        predicted_emotion = map_to_emotion(scores['roberta_pos'], scores['roberta_neg'], scores['roberta_neu'])
        
        audio_path = generate_tts(text, emotion, voice_id)
        
        new_audio = {
            "text": text,
            "emotion": emotion,
            "path": audio_path
        }
        result = audio_collection.insert_one(new_audio)
        
        return jsonify({'audio_path': audio_path, 'predicted_emotion': predicted_emotion, 'id': str(result.inserted_id)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/audio/<path:filename>')
def get_audio(filename):
    return send_file(os.path.join('static', filename), mimetype='audio/mp3')

if __name__ == '__main__':
    app.run(debug=True)







