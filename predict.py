import librosa
import numpy as np
from tensorflow.keras.models import load_model


MODEL_DIR = ''
AUDIO_PATH = ''
JSON_PATH = "data/save.json"

N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048
SAMPLE_RATE = 22050
TRACK_DURATION = 30
model = load_model(MODEL_DIR)


def load_audio(audio_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    SAMPLE_RATE = 22050
    TRACK_DURATION = 30
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    signal, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)
    start = samples_per_segment * 4
    finish = start + samples_per_segment
    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                hop_length=hop_length).T
    return mfcc


data = load_audio(AUDIO_PATH)


def predict(model, X, y):
    X = X[np.newaxis, ...]
    prediction = model.predict(X)
    predicted_index = np.argmax(prediction, axis=1)
    print("Target: {}, Predicted label: {}".format(y, predicted_index))


predict(model, data, 6)
