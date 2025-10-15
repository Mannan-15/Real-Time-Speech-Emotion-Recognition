import pickle
import os

import librosa
import numpy as np
from flask import Flask, render_template, redirect, request
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
model_dict = pickle.load(open('stackclassifier.pkl', 'rb'))
model = model_dict['classifier']
scaler = pickle.load(open('scaler.pkl', 'rb'))


# data augmentation
def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    noised = data + noise_amp * np.random.normal(size=data.shape[0])
    return noised


def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)


def pitch(data, sr, rate=0.7):
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=rate)


def shift(data):
    range = int(np.random.uniform(low=-5, high=5) * 10)
    return np.roll(data, range)


# Features Extraction
def extract_features(data, sr):
    result = np.array([])
    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(data).T, axis=0)
    result = np.hstack((result, zcr))

    # chroma stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # mfcc
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, mfcc))

    # rms value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # mel spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, mel))

    return result


def get_features(audio):
    arr, sr = librosa.load(audio, duration=2.5, offset=0.6)
    result = np.array(extract_features(arr, sr))

    noisy = noise(arr)
    result = np.vstack((result, extract_features(noisy, sr)))

    pitchy = pitch(stretch(arr), sr)
    result = np.vstack((result, extract_features(pitchy, sr)))

    return result


@app.route('/')
def index():  # put application's code here
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'audio' not in request.files:
            return redirect(request.url)

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return redirect(request.url)

        feature = get_features(audio_file)
        scaledData = scaler.transform(feature)
        predictResult = model.predict(scaledData)[0]
        predictedScore = max(model.predict_proba(scaledData)[0])*100
        labels = {0: 'angry', 1 : 'calm', 2 : 'disgust', 3 : 'fear', 4 : 'happy', 5 : 'neutral', 6 : 'sad',
        7 : 'surprise'}
        predictResult = labels.get(int(predictResult), "unknown")
        return render_template('index.html', prediction =f"{predictResult.title()} with confidence: {predictedScore : .2f}%")


if __name__ == '__main__':
    app.run(debug = True)
