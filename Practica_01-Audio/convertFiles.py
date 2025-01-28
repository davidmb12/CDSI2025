import ffmpeg.audio
from pydub import AudioSegment
import ffmpeg
import os
from pathlib import Path
import subprocess
def convertFiles(directoryPath,outputDirectory,fromExt,toExt):
    audio_files = [f for f in os.listdir(directoryPath) if f.endswith(fromExt)]
    outputDir = outputDirectory
    for file in audio_files:
        print(file)
        subprocess.run(
            ["ffmpeg", "-i", directoryPath+file, outputDir+file.replace(fromExt,toExt)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        #Cargar el archivo y exportarlo con el formato establecido
        # audio = AudioSegment.from_file('./AUDIO/Cielo-Higuera/Practica-01_Audio/Cielo-01_01.m4a',format="m4a")
        # audio.export('./AUDIO/Cielo-Higuera/Practica-01_Audio/Cielo-01_01.m4a',format="wav")
        