# train_model.py

import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from cnn_model import create_cnn_model

def load_data(data_folder):
    X = []
    y = []
    for file in os.listdir(data_folder):
        if file.endswith('.npy'):
            mfccs = np.load(os.path.join(data_folder, file))
            if mfccs.shape == (13, 32):  # Ensure the shape is as expected
                mfccs = np.expand_dims(mfccs, axis=-1)
                X.append(mfccs)
                # Assuming the file names contain labels, e.g., 'label_filename.npy'
                label = file.split('_')[0]
                y.append(label)
            else:
                print(f"Skipping file {file} due to unexpected shape: {mfccs.shape}")
    
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No valid data found in the specified folder.")
    
    X = np.array(X)
    y = np.array(y)
    return X, y

data_folder = 'E:\Voice_Coach\Wav\prepo'  # Replace with your actual path
X, y = load_data(data_folder)
print("Data shape:", X.shape)
print("Labels shape:", y.shape)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))
y_one_hot = to_categorical(y_encoded, num_classes=num_classes)

X_train, X_val, y_train, y_val = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

model = create_cnn_model(num_classes)
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
model.save('voice_coach_model.keras')  # Save using the .keras extension
