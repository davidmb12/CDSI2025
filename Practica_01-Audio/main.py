from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from audioLoading import loadAudios
from pathlib import Path
from scipy.io import wavfile
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import matplotlib.pyplot as plt, IPython.display as ipd
import librosa, librosa.display
import math
import ffmpeg
from convertFiles import convertFiles
from audioManipulation import timeStretch,pitchShift,reverse,denoise
audioToIdx ={
    'AdriaM':0,
    'AleM':1,
    'BetoM':2,
    'Cielo':3,
    'Daniel':4,
    'Erik':5,
    'Irma':6,
    'LuisG':7,
    'Maria':8,
    'Mariana':9,
    'Mario':10,
    'Marlon':11,
    'MauM':12,
    'Sergio':13,
    'Vanessa':14,
    'Nula':15,
}
def audioFileNameToIdx(audioFileName):
    return audioToIdx[audioFileName]

def main(directory_path):
    audios = loadAudios(f"{directory_path}*")
    audios_features = []
    for audio_signal,filename in audios:
        print(audio_signal[0].dtype)
        features = extractFeatures(audio_signal[0])
        label = audioFileNameToIdx(str.split(filename,'-')[0])
        audios_features.append([label] + features)
    
    df = pd.DataFrame(audios_features,columns=["Label","ZeroCrossingRate","Tempo","MFCC"])
    df.to_csv("audio_features.csv",index=False)
    print("Features guardadas en audio_features.csv")

def extractFeatures(signal):
    return [
        librosa.feature.spectral_contrast(y=signal)[0,0],
        librosa.feature.chroma_cens(y=signal)[0,0],
        librosa.feature.mfcc(y=signal)[0,0],
        librosa.feature.zero_crossing_rate(y=signal)[0,0],
        librosa.feature.rms(y=signal)[0,0],
        librosa.feature.melspectrogram(y=signal)
    ]
def sandbox():
    features = [
        np.mean(librosa.feature.mfcc(y=audio_signal[0], sr=audio_signal[1], n_mfcc=13), axis=1)[0],
        np.var(librosa.feature.mfcc(y=audio_signal[0], sr=audio_signal[1], n_mfcc=13), axis=1)[0],
        np.mean(librosa.feature.spectral_contrast(y=audio_signal[0],sr=audio_signal[1],n_fft=256)[0,0]),
        np.mean(librosa.feature.spectral_centroid(y=audio_signal[0],sr=audio_signal[1])[0,0]),
        np.mean(librosa.feature.spectral_bandwidth(y=audio_signal[0],sr=audio_signal[1])[0,0]),
        np.mean(librosa.feature.chroma_stft(y=audio_signal[0],sr=audio_signal[1])[0,0]),
        np.mean(librosa.feature.zero_crossing_rate(y=audio_signal[0])[0,0]),
        np.mean(librosa.feature.rms(y=audio_signal[0])[0,0]),
        np.mean(librosa.feature.melspectrogram(y=audio_signal[0],sr=audio_signal[1])[0,0]),
        np.mean(librosa.yin(y=audio_signal[0], fmin=50, fmax=500))
    ]
def plot(rows,cols,audios):
    pass

def prepareData():
    df = pd.read_csv("audio_features.csv")
    
    #Dividir caracteristicas (x) y labels (y)
    x=df[["ZeroCrossingRate","Tempo","MFCC"]]
    y=df["Label"]
    DecisionTreeClassifier()
    #Dividir los datos en conjuntos de entrenamiento y prueba
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    return (x_train,x_test,y_train,y_test)
MinMaxScaler()

# convertFiles('./AUDIO/Luis-Garcia/Practica-01_Audio/','./AUDIO/All-Audios/','ogg','wav')
# convertFiles('./AUDIO/Luis-Garcia/Practica-01_Audio/','./AUDIO/All-Audios/','m4a','wav')
# convertFiles('./AUDIO/Cielo-Higuera/Practica-01_Audio/','./AUDIO/All-Audios/','m4a','wav')
# convertFiles('./AUDIO/Cielo-Higuera/Practica-01_Audio/','./AUDIO/All-Audios/','m4a','wav')
# main('./AUDIO/All-Audios/')
#print(prepareData())
denoise('./AUDIO/All-Audios/Metodo-01/','./AUDIO/All-Audios/Metodo-01_Denoised/')



#convertFiles('./AUDIO/All-Audios/','./AUDIO/All-Audios/','opus','wav')
