# backend/preprocess.py
import os
import librosa
import numpy as np

DATASET_PATH = "dataset/ravdess/"
emotion_map = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgusted", "08": "surprised"
}

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def load_dataset():
    features, labels = [], []
    for actor_folder in os.listdir(DATASET_PATH):
        actor_path = os.path.join(DATASET_PATH, actor_folder)
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                emotion_code = file.split("-")[2]
                if emotion_code in emotion_map:
                    label = emotion_map[emotion_code]
                    mfcc = extract_features(os.path.join(actor_path, file))
                    features.append(mfcc)
                    labels.append(label)
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    X, y = load_dataset()
    np.save("data/X_mfcc.npy", X)
    np.save("data/y_labels.npy", y)
