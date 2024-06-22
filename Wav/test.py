import pyaudio
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from matplotlib.animation import FuncAnimation
import threading

# Parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SPECTROGRAM_SHAPE = (128, 128)  # Example shape, adjust as needed

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Function to get audio data
def get_audio_data():
    audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
    return audio_data

# Function to extract spectrogram
def extract_spectrogram(audio_data, sr=RATE):
    S = librosa.feature.melspectrogram(y=audio_data.astype(float), sr=sr, n_mels=SPECTROGRAM_SHAPE[0])
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db_resized = librosa.util.fix_length(S_db, SPECTROGRAM_SHAPE[1], axis=1)
    return S_db_resized

# Build CNN Model
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')  # Adjust the output layer to match your number of classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Example input shape for spectrograms
input_shape = (*SPECTROGRAM_SHAPE, 1)
model = build_model(input_shape)

# Training the model (Example)
def train_model(model, X_train, y_train, epochs=10):
    X_train = np.expand_dims(X_train, axis=-1)  # Ensure correct shape for CNN
    model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)

# Placeholder for training data
# Replace with real spectrogram data and labels
X_train = np.random.rand(100, *SPECTROGRAM_SHAPE)  # Replace with real spectrogram data
y_train = np.random.randint(0, 10, 100)  # Replace with real labels

train_model(model, X_train, y_train, epochs=10)

# Real-time prediction
def predict(model, spectrogram):
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
    prediction = model.predict(spectrogram)
    return np.argmax(prediction)

# Real-time processing loop
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'b-')

def init():
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)  # Adjust based on the number of classes
    return ln,

def update(frame):
    xdata.append(len(xdata) / RATE)
    ydata.append(frame)
    ln.set_data(xdata, ydata)
    if len(xdata) > RATE * 10:
        xdata.pop(0)
        ydata.pop(0)
    ax.set_xlim(max(0, len(xdata) / RATE - 10), len(xdata) / RATE)
    return ln,

def process_audio():
    while True:
        audio_data = get_audio_data()
        S_db = extract_spectrogram(audio_data)
        prediction = predict(model, S_db)
        update(prediction)

def animate(i):
    audio_data = get_audio_data()
    S_db = extract_spectrogram(audio_data)
    prediction = predict(model, S_db)
    update(prediction)
    return ln,

ani = FuncAnimation(fig, animate, init_func=init, blit=True)

# Start the real-time processing in a separate thread
thread = threading.Thread(target=process_audio)
thread.start()

plt.show()

# Clean up
stream.stop_stream()
stream.close()
p.terminate()
    