# preprocess_audio.py

import librosa
import numpy as np
import os

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs

def save_preprocessed_data(audio_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(audio_folder):
        if filename.endswith('.wav'):
            mfccs = preprocess_audio(os.path.join(audio_folder, filename))
            np.save(os.path.join(output_folder, filename.replace('.wav', '.npy')), mfccs)

if __name__ == '__main__':
    audio_folder = 'E:\Voice_Coach\Wav\Audio'
    output_folder = 'E:\Voice_Coach\Wav\prepo'
    save_preprocessed_data(audio_folder, output_folder)
