import librosa
import numpy as np
from tensorflow.keras.models import load_model


MODEL_DIR = ''
AUDIO_PATH = ''

N_FFT = 2048
N_MFCC = 13
HOP_LENGTH = 512
SAMPLE_RATE = 44100
TRACK_DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def load_audio(audio_path, num_mfcc, n_fft, hop_length, num_segments):
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    start = samples_per_segment * 4
    finish = start + samples_per_segment
    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sr, n_mfcc=num_mfcc, n_fft=n_fft,
                                hop_length=hop_length).T
    return mfcc


def predict(model, X, y):
    X = X[np.newaxis, ...]
    prediction = model.predict(X)
    predicted_index = np.argmax(prediction, axis=1)
    print("Target: {}, Predicted label: {}".format(y, predicted_index))


model = load_model(MODEL_DIR)
data = load_audio(AUDIO_PATH, N_MFCC, N_FFT, HOP_LENGTH, 10)
predict(model, data, 6)
