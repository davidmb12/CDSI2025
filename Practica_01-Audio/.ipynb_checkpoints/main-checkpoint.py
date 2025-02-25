from audioLoading import loadAudios
from pathlib import Path
from scipy.io import wavfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, IPython.display as ipd
import librosa, librosa.display
import math
import ffmpeg
from convertFiles import convertFiles

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
        librosa.feature.zero_crossing_rate(signal)[0,0],
        librosa.feature.fourier_tempogram(y=signal)[0,0],
        librosa.feature.mfcc(y=signal)[0,0],
    ]

def plot(rows,cols,audios):
    pass

# directories = {'./AUDIO/David-Murillo/Practica-01_Audio/':'ogg','./AUDIO/Mario-Lizarraga/Practica-01_Audio/':'mp3'}
# main('./AUDIO/All-Audios/')
convertFiles('./AUDIO/Cielo-Higuera/Practica-01_Audio/','m4a','wav')
