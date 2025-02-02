from pathlib import Path
from scipy.io import wavfile
import librosa, librosa.display

import librosa
def loadAudios(directoryPath,fileExtension ="wav"):
    audio_data = [[]]
    try:
        audio_data = [[librosa.load(p),p.name] for p in Path().glob(directoryPath)]
    except:
        print("Tipo de archivo no admitido")
    return audio_data
