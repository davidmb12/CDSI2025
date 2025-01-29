from audioLoading import loadAudios
import librosa
import soundfile as sf
import os

def timeStretch(inputFile,fromExt,toExt="wav"):
    audio,sr =librosa.load(f"{inputFile}.{fromExt}")
    audioStretched = librosa.effects.time_stretch(y=audio,rate=1.5)
    sf.write(f"{inputFile}-stretched.{toExt}",audioStretched,sr)
    print(audio)