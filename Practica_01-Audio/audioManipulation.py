from audioLoading import loadAudios
import librosa
import soundfile as sf
import os
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def timeStretch(directoryPath,outputDirectory):
    audio_files = [f for f in os.listdir(directoryPath) ]
    for file in audio_files:
        print(file)
        audio,sr =librosa.load(f"{directoryPath}{file}")
        audioStretched = librosa.effects.time_stretch(y=audio,rate=1.5)
        sf.write(f"{outputDirectory}TimeStretched_{file}",audioStretched,sr)
        
def pitchShift(directoryPath,outputDirectory):
    audio_files = [f for f in os.listdir(directoryPath) if not f.__contains__("TimeStretched")]
    for file in audio_files:
        print(file)
        audio,sr =librosa.load(f"{directoryPath}{file}")
        audioStretched = librosa.effects.pitch_shift(y=audio,sr=sr,n_steps=4)
        sf.write(f"{outputDirectory}PitchShifted_{file}",audioStretched,sr)
        
def reverse(directoryPath,outputDirectory):
    audio_files = [f for f in os.listdir(directoryPath) if not f.__contains__("TimeStretched") or f.__contains__("PitchShifted")]
    for file in audio_files:
        print(file)
        audio,sr =librosa.load(f"{directoryPath}{file}")
        audioReversed= audio[::-1]
        sf.write(f"{outputDirectory}Reversed{file}",audioReversed,sr)
        
def denoise(inputFile,directoryPath,outputDirectory,window_size=5):
    y,sr = librosa.load(f"{directoryPath}{inputFile}")
    y_denoised =np.convolve(y,np.ones(window_size)/window_size,mode ='same')
    sf.write(f"{outputDirectory}{inputFile}",y_denoised,sr)