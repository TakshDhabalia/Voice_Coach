import wave
import numpy as np
import pandas as pd
import librosa
def wav_to_csv(wav_file, csv_file):
    # Open the WAV file
    with wave.open(wav_file, 'r') as wav:
        # Get WAV file parameters
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        framerate = wav.getframerate()
        n_frames = wav.getnframes()

        # Read the audio frames
        frames = wav.readframes(n_frames)
        
        # Convert frames to numpy array
        if sampwidth == 1:
            dtype = np.uint8  # 8-bit audio
        elif sampwidth == 2:
            dtype = np.int16  # 16-bit audio
        elif sampwidth == 4:
            dtype = np.int32  # 32-bit audio
        else:
            raise ValueError("Unsupported sample width")
        
        audio_data = np.frombuffer(frames, dtype=dtype)
        
        # Reshape data to separate channels
        audio_data = audio_data.reshape(-1, n_channels)

        # Create a pandas DataFrame
        df = pd.DataFrame(audio_data)

        # Save the DataFrame to a CSV file
        df.to_csv(csv_file, index=False, header=False)

# Example usage
wav_to_csv('VK3_Sargam.wav', '1.csv')
