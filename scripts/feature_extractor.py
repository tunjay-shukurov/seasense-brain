import numpy as np
from obspy import read
import pandas as pd

def extract_features_from_signal(data, fs):
    """
    data: 1D numpy array, raw or filtered signal
    fs: sampling frequency (Hz)

    returns: feature vector (list of floats)
    """
    n = len(data)
    duration = n / fs

    # Basic statistics
    mean_val = np.mean(data)
    std_val = np.std(data)
    max_val = np.max(data)
    min_val = np.min(data)
    rms = np.sqrt(np.mean(data**2))

    # Advanced statistics
    kurtosis = pd.Series(data).kurt() if n > 3 else 0
    skewness = pd.Series(data).skew() if n > 3 else 0

    # Zero Crossing Rate
    zcr = ((data[:-1] * data[1:]) < 0).sum() / n

    # FFT
    freq = np.fft.rfftfreq(n, d=1/fs)
    Y = np.abs(np.fft.rfft(data))

    # Spectral features
    spectral_centroid = np.sum(freq * Y) / np.sum(Y) if np.sum(Y) > 0 else 0
    peak_freq = freq[np.argmax(Y)] if len(Y) > 0 else 0

    # Band energies (example: 1-5 Hz, 5-10 Hz)
    band_energy_1_5Hz = np.sum(Y[(freq >= 1) & (freq <= 5)])
    band_energy_5_10Hz = np.sum(Y[(freq > 5) & (freq <= 10)])

    features = [
        duration,
        mean_val,
        std_val,
        max_val,
        min_val,
        rms,
        kurtosis,
        skewness,
        zcr,
        spectral_centroid,
        peak_freq,
        band_energy_1_5Hz,
        band_energy_5_10Hz
    ]

    return features

def extract_features_from_sac(file_path):
    """
    file_path: SAC file path
    returns: features list (numeric)
    """
    st = read(file_path)
    tr = st[0]
    data = tr.data.astype(np.float32)
    fs = tr.stats.sampling_rate
    return extract_features_from_signal(data, fs)
