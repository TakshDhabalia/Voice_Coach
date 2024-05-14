import pyaudio
import numpy as np
from crepe import predict

# Constants
CHUNK_SIZE = 1024
SAMPLE_RATE = 16000  # Model expects 16 kHz sample rate
MODEL_CAPACITY = 'full'  # Choose model capacity (e.g., 'tiny', 'small', 'medium', 'large', 'full')        : chose full for the default purposes 
STEP_SIZE = 10  # Step size for pitch estimation in milliseconds , 100 doesnt work well not aware for the resons 

# Initialize PyAudio
audio_interface = pyaudio.PyAudio()
stream = audio_interface.open(format=pyaudio.paInt16,
                              channels=1,
                              rate=SAMPLE_RATE,
                              input=True,
                              frames_per_buffer=CHUNK_SIZE)

print("CREPE: Starting real-time pitch estimation...")

try:
    while True:
        # Read audio input from the microphone , default 0 ?
        audio_data = np.frombuffer(stream.read(CHUNK_SIZE), dtype=np.int16)

        # Perform pitch estimation
        time, frequency, confidence, _ = predict(audio_data, SAMPLE_RATE,
                                                 model_capacity=MODEL_CAPACITY,
                                                 step_size=STEP_SIZE,
                                                 verbose=False)

        # Print the estimated pitch (Hz)
        if confidence.max() > 0.5:  # Adjust confidence threshold as needed
            print("Estimated Pitch (Hz):", frequency[np.argmax(confidence)] , "confidence found was : ", confidence[np.argmax(confidence)]  )
        else:
            print("No confident pitch estimate.")

except KeyboardInterrupt:
    print("CREPE: Stopping real-time pitch estimation...")

finally:
    # Clean up
    stream.stop_stream()
    stream.close()
    audio_interface.terminate()
