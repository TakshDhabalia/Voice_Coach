
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split



# Example sine wave filenames and their corresponding labels (frequency ranges)
sine_wave_files = [
  r'C:\Users\bhate\OneDrive\Documents\lstm\SineWaveFiles\sine_wave_high.wav',
    r'C:\Users\bhate\OneDrive\Documents\lstm\SineWaveFiles\sine_wave_low.wav',
    r'C:\Users\bhate\OneDrive\Documents\lstm\SineWaveFiles\sine_wave_medium.wav'
]
labels = [2, 0, 1]  # corresponding labels for high, low, and medium frequencies

# Parameters
sr = 22050  # Sampling rate
n_mels = 64  # Number of Mel bands
fmax = sr / 2  # Maximum frequency

# Function to load and preprocess audio files
def load_and_preprocess(filename, sr, n_mels, fmax):
    y, sr = librosa.load(filename, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

# Load and preprocess all sine wave files
X = []
for file in sine_wave_files:
    mel_spectrogram = load_and_preprocess(file, sr, n_mels, fmax)
    X.append(mel_spectrogram)

# Convert to numpy arrays
X = np.array(X)
y = np.array(labels)

# Expand dimensions to match the input shape expected by Keras (e.g., add a channel dimension)
X = np.expand_dims(X, axis=-1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a simple CNN model
model = keras.Sequential([
    keras.layers.Input(shape=x_train.shape[1:]),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(3, activation='softmax')  # Assuming 3 classes for 3 frequency ranges
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

# Evaluate the model
model.evaluate(x_test, y_test)
