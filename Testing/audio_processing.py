import numpy as np
import librosa

# Function to get audio data
def get_audio_data(stream, chunk=1024):
    audio_data = np.frombuffer(stream.read(chunk), dtype=np.int16)
    audio_data = audio_data.astype(np.float32) / 32768.0  # Convert to floating-point and normalize
    return audio_data

# Function to extract Mel-spectrogram
def extract_mel_spectrogram(audio, sr=44100, n_mels=128, shape=(128, 128)):
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db_resized = librosa.util.fix_length(S_db, shape[1], axis=1)
    return S_db_resized
