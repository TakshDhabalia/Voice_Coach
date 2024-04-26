#proprocessing the audio file 
"""
1. conver audio into digital format - mp3  , or WAV maybe 

2. noise filteeration for basic preprocessing 

3. PITCH [FRAME WISE or BULK FRAME SLIDING WINDOWish APPROACH ]
    3.1  FFT - fast Furior Transform 
    3.2  Harmonic Product Transform 

    3.3 - Autocorrelation

4.Spectral Features 

    4.1 Mel-Frequency Cepstral Coefficients (MFCCs)
    4.2 Spectral Centroid, Spectral Flux
    4.3 Spectral Roll-off

5. Temporal Features
    5,1 Zero Crossing Rate (ZCR) 
    5.2 Temporal Centroid
    5.3 Temporal Dynamics [MAYBE !]

6. Perceptual Features
    6.1 Loudness 
    6.2 Timbre 
    6.3 Amplitute 
        6.3.1 Harmonics-to-Noise Ratio (HNR)

7. Emotion 


"""