from audioLoading import loadAudios
import librosa
import soundfile as sf
import os
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr
from pydub import AudioSegment
def timeStretch(directoryPath,outputDirectory,fromExt = "wav",toExt="wav",rate=1.2):
    audio_files = [f for f in os.listdir(directoryPath) ]
    for file in audio_files:
        audio,sr =librosa.load(f"{directoryPath}{file}")
        audioStretched = librosa.effects.time_stretch(y=audio,rate=rate)
        sf.write(f"{outputDirectory}{str.replace(file,f".{fromExt}",'')}_TimeStretched02.{toExt}",audioStretched,sr)
        
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


def preprocess_audio(directoryPath,output_path,singleAudio = False):
    audio_files = [f for f in os.listdir(directoryPath)]
    if not singleAudio:
        for file in audio_files:
            y,sr = librosa.load(f"{directoryPath}{file}",sr = 16000)
            y = nr.reduce_noise(y=y,sr=sr)
            y,_ = librosa.effects.trim(y)
            fileName = str.split(file,".")
            sf.write(f"{output_path}{fileName[0]}.wav",y,sr)
            audio = AudioSegment.from_wav(f"{output_path}{fileName[0]}.wav")
            normalized_audio = audio.apply_gain(-audio.dBFS)
            normalized_audio.export(f"{output_path}{fileName[0]}.wav",format="wav")
    
        


def renameFiles(directoryPath, prefix='', suffix='',start_number=1,add_numbering=False,keepName=True):
    try:
        files = sorted(os.listdir(directoryPath))  # Sort to ensure numbering is consistent
        for index, filename in enumerate(files, start=start_number):
            file_path = os.path.join(directoryPath, filename)
            if os.path.isfile(file_path):
                name, ext = os.path.splitext(filename)
                new_name = f"{prefix}{name}{index if add_numbering else ''}{suffix}{ext}"
                new_path = os.path.join(directoryPath, new_name)
                os.rename(file_path, new_path)
                print(f"Renamed: {filename} -> {new_name}")
    except Exception as e:
        print(f"Error: {e}")
