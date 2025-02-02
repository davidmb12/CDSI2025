import os
import pandas as pd
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Kept RandomForest as requested
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

import pyaudio
import wave
import joblib

# Paths
DATASET_PATH = r'C:\Users\ADMIN\Desktop\Metodo1'
NOISE_PATH = r'C:\Users\ADMIN\Desktop\Metodo1\noise'  # Folder with noise samples
OUTPUT_FOLDER = "processed_audio_01/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Collect dataset files
data = []

for speaker in os.listdir(DATASET_PATH):
    speaker_folder = os.path.join(DATASET_PATH, speaker)

    if os.path.isdir(speaker_folder):
        for file in os.listdir(speaker_folder):
            if file.endswith(".wav"):  
                file_path = os.path.join(speaker_folder, file)
                label = "unknown" if "noise" in speaker.lower() else speaker
                data.append({"filename": file, "speaker": label, "filepath": file_path})

df = pd.DataFrame(data)

# Preprocessing Function (Noise Reduction, Silence Trimming, Normalization)
def preprocess_audio(file_path, output_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)

        # If the audio is silent, do not process it
        if np.max(np.abs(y)) < 0.01:
            return None  

        y = nr.reduce_noise(y=y, sr=sr)
        y, _ = librosa.effects.trim(y)
        y = librosa.util.normalize(y)  # Normalization without ffmpeg
        sf.write(output_path, y, sr)
        return output_path  
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None  

df["processed_filepath"] = df["filepath"].apply(lambda x: preprocess_audio(x, os.path.join(OUTPUT_FOLDER, os.path.basename(x).replace(".wav", "_clean.wav"))))

df.to_csv("audio_dataset_preprocessed.csv", index=False)

# Data Augmentation Functions
def add_noise(y, noise_level=0.01):
    noise = np.random.randn(len(y)) * noise_level
    return y + noise

def change_speed(y, sr, speed_factor=1.1):
    return librosa.effects.time_stretch(y, rate=speed_factor)

def change_pitch(y, sr, n_steps=3):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

def time_shift(y, shift_max=0.3):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)

def augment_audio(y, sr):
    return [
        add_noise(y),
        change_speed(y, sr, speed_factor=random.uniform(0.8, 1.2)),
        change_pitch(y, sr, n_steps=random.choice([-3, -2, 2, 3])),
        time_shift(y),
    ]

# Apply Data Augmentation
augmented_data = []

for _, row in df.iterrows():
    y, sr = librosa.load(row["processed_filepath"], sr=None)
    augmented_audios = augment_audio(y, sr)
    
    for i, aug_y in enumerate(augmented_audios):
        aug_file = row["processed_filepath"].replace(".wav", f"_aug{i}.wav")
        sf.write(aug_file, aug_y, sr)
        augmented_data.append({"filename": os.path.basename(aug_file), "speaker": row["speaker"], "filepath": aug_file})

df_aug = pd.DataFrame(augmented_data)
df = pd.concat([df, df_aug], ignore_index=True)

df.to_csv("audio_dataset_augmented.csv", index=False)

# Feature Extraction
def extract_mfcc(y, sr, num_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
    return np.mean(mfccs, axis=1)

def extract_pitch(y, sr):
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    return np.mean(pitch[pitch > 0])

def extract_spectral_centroid(y, sr):
    return np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

def extract_zcr(y):
    return np.mean(librosa.feature.zero_crossing_rate(y))

def extract_spectral_bandwidth(y, sr):
    return np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # If audio is silent, return None
    if np.max(np.abs(y)) < 0.01:
        return None  

    mfcc_values = extract_mfcc(y, sr, num_mfcc=13)

    features = {f"mfcc_{i+1}": mfcc_values[i] for i in range(13)}
    features["pitch"] = extract_pitch(y, sr)
    features["spectral_centroid"] = extract_spectral_centroid(y, sr)
    features["zcr"] = extract_zcr(y)
    features["spectral_bandwidth"] = extract_spectral_bandwidth(y, sr)
    
    return features

features_list = [extract_features(fp) for fp in df["filepath"] if extract_features(fp) is not None]
features_df = pd.DataFrame(features_list)

df = pd.concat([df, features_df], axis=1)
df.to_csv("audio_features.csv", index=False)

# Model Training
df = pd.read_csv("audio_features.csv")

X = df.drop(columns=["filename", "speaker", "filepath", "processed_filepath"])
y = df["speaker"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100))  # Kept RandomForest
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

joblib.dump(model, "speaker_recognition_model.pkl")

# Real-Time Speaker Identification
def record_audio(filename="test.wav", duration=3, sr=16000):
    print("Recording...")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sr, input=True, frames_per_buffer=1024)
    frames = []

    for _ in range(0, int(sr / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sr)
    wf.writeframes(b"".join(frames))
    wf.close()

    print("Recording finished.")

def recognize_speaker():
    record_audio()
    features = extract_features("test.wav")

    if features is None:
        print("No speech detected.")
        return  

    feature_df = pd.DataFrame([features], columns=X_train.columns)

    model = joblib.load("speaker_recognition_model_01.pkl")
    probs = model.predict_proba(feature_df)
    speaker_labels = model.classes_

    max_prob = max(probs[0])
    best_speaker = speaker_labels[np.argmax(probs[0])]

    if max_prob < 0.2:  
        best_speaker = "unknown"

    print(f"Identified Speaker: {best_speaker} (Confidence: {max_prob:.2f})")

recognize_speaker()