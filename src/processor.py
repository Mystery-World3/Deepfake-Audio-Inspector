import librosa
import numpy as np

def extract_audio_features(file_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=None)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    
    return mfccs_processed