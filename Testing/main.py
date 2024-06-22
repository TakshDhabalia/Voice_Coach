import threading
import numpy as np
import pyaudio
import librosa
from audio_processing import get_audio_data, extract_mel_spectrogram
from model import build_cnn_model, train_model
from real_time_feedback import process_audio

# Parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SPECTROGRAM_SHAPE = (128, 128)  # Example shape, adjust as needed
REFERENCE_AUDIO_PATH = 'E:\Voice_Coach\Wav\Audio\VK_Palta221221.wav'

# Load reference audio
reference_audio, sr = librosa.load(REFERENCE_AUDIO_PATH, sr=RATE)
reference_spectrogram = extract_mel_spectrogram(reference_audio, sr=sr)

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Example input shape for spectrograms
input_shape = (*SPECTROGRAM_SHAPE, 1)
model = build_cnn_model(input_shape)

# Create training data by slightly augmenting the reference spectrogram (for example purposes)
X_train = np.array([reference_spectrogram for _ in range(100)])  # Replace with actual data augmentation
y_train = np.array([0 for _ in range(100)])  # Labels; adjust as needed for your use case

# Train the model
train_model(model, X_train, y_train, epochs=10)

# Save the trained model (optional)
model.save('voice_coach_model.h5')

# Start the real-time processing in the main thread
process_audio(model, stream, RATE, reference_spectrogram)

# Clean up
stream.stop_stream()
stream.close()
p.terminate()
