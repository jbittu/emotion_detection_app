from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import os
import joblib
import tensorflow as tf
import pickle
import uuid

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
svm_model = joblib.load("models/svm_model.pkl")
rf_model = joblib.load("models/rf_model.pkl")
cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
lstm_model = tf.keras.models.load_model("models/lstm_model.h5")
gru_model = tf.keras.models.load_model("models/gru_model.h5")

# Load label encoder
with open("models/rf_label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Check file format
        if not file.filename.endswith(".wav"):
            return JSONResponse({"error": "Only .wav files are supported."}, status_code=400)

        # Save uploaded WAV file
        uid = str(uuid.uuid4())
        temp_wav = f"temp_{uid}.wav"
        with open(temp_wav, "wb") as f:
            f.write(await file.read())

        # Extract features
        mfcc = extract_features(temp_wav)
        mfcc_input = mfcc.reshape(1, -1)
        mfcc_seq = mfcc.reshape(1, 40, 1)

        # Predict with models
        pred_svm = le.inverse_transform([svm_model.predict(mfcc_input)[0]])[0]
        pred_rf = le.inverse_transform([rf_model.predict(mfcc_input)[0]])[0]
        pred_cnn = le.inverse_transform([np.argmax(cnn_model.predict(mfcc_seq))])[0]
        pred_lstm = le.inverse_transform([np.argmax(lstm_model.predict(mfcc_seq))])[0]
        pred_gru = le.inverse_transform([np.argmax(gru_model.predict(mfcc_seq))])[0]

        return JSONResponse({
            "svm_prediction": pred_svm,
            "rf_prediction": pred_rf,
            "cnn_prediction": pred_cnn,
            "lstm_prediction": pred_lstm,
            "gru_prediction": pred_gru
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
