# real_time_inference.py

import pyaudio
import numpy as np
import librosa
import tensorflow as tf

# Function to perform real-time inference
def real_time_inference():
    # Code for recording audio
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 5

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Convert audio data to numpy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=RATE, n_mfcc=13)

    # Adjust MFCC shape
    mfccs_reshaped = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)

    # Load the pre-trained model
    model = tf.keras.models.load_model('voice_coach_model.keras')

    # Perform prediction
    predictions = model.predict(mfccs_reshaped)

    # Assuming predictions are a numpy array or similar
    predicted_class = np.argmax(predictions)

    return predicted_class
