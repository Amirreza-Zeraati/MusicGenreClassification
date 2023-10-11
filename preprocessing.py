import os
import math
import json
import librosa

DATA_PATH = 'data/genre/genres_original'
JSON_PATH = ''
SAMPLE_RATE = 22050
TRACK_DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def get_mfccs(data_path, json_path, num_mfcc=13, n_fft=1024, hop_length=256, num_segments=5, save=False, returnn=False):
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(data_path)):
        if dirpath is not data_path:

            semantic_label = dirpath.split("\\")[-1]
            print(semantic_label)
            data["mapping"].append(semantic_label)

            for file in filenames:
                signal, sr = librosa.load(os.path.join(dirpath, file), sr=SAMPLE_RATE)

                for d in range(num_segments):
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sr, n_mfcc=num_mfcc, 
                                                n_fft=n_fft, hop_length=hop_length).T

                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)

    if save:
        with open(json_path, 'w') as fp:
            json.dump(data, fp, indent=4)

    if returnn:
        return data


if __name__ == "__main__":
    get_mfccs(DATA_PATH, JSON_PATH)
