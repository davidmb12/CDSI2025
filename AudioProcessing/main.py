import librosa
from IPython.display import Audio

#1 -- Librosa
audio, sr = librosa.load("DanielCaesarSong.mp3")
tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
print(f"Tempo: {tempo}")


# #2--soundfile

# import soundfile as sf

# data,samplerate = sf.read("sound_01.wav")
# sf.write("sound_01.wav",data , samplerate)

# # 3-- Wave

# import wave

# with wave.open("sound_01.wav", "rb") as wav_file:
#     print("Channels:", wav_file.getnchannels())
#     print("Sample width:", wav_file.getsampwidth())

# # 4 -- PyAudio -- Capture audio in realtime

# import pyaudio

# p = pyaudio.PyAudio()
# print("Default input device info:", p.get_default_input_device_info())


# # 5 -- SciPy

# from scipy.io import wavfile

# rate, data = wavfile.read("sound_01.wav")
# print(f"Sample rate: {rate}, Data shape: {data.shape}")

# # TorchAudio

# # Essentia
