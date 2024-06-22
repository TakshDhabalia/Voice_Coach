from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np 
# Build CNN Model
def build_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='linear')  # Adjust the output for regression or classification
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the model (Example)
def train_model(model, X_train, y_train, epochs=10):
    X_train = np.expand_dims(X_train, axis=-1)  # Ensure correct shape for CNN
    model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)

# Load the trained model (optional)
def load_trained_model(model_path='voice_coach_model.h5'):
    return load_model(model_path)
