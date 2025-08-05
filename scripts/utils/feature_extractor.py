import numpy as np
import pandas as pd

def extract_features_from_signal(data: np.ndarray, fs: float) -> list:
    n = len(data)
    duration = n / fs
    mean_val = np.mean(data)
    std_val = np.std(data)
    max_val = np.max(data)
    min_val = np.min(data)
    rms = np.sqrt(np.mean(data**2))
    kurtosis = pd.Series(data).kurt() if n > 3 else 0
    skewness = pd.Series(data).skew() if n > 3 else 0
    zcr = ((data[:-1] * data[1:]) < 0).sum() / n
    freq = np.fft.rfftfreq(n, d=1/fs)
    Y = np.abs(np.fft.rfft(data))
    spectral_centroid = np.sum(freq * Y) / np.sum(Y) if np.sum(Y) > 0 else 0
    peak_freq = freq[np.argmax(Y)] if len(Y) > 0 else 0
    band_energy_1_5Hz = np.sum(Y[(freq >= 1) & (freq <= 5)])
    band_energy_5_10Hz = np.sum(Y[(freq > 5) & (freq <= 10)])

    return [
        duration, mean_val, std_val, max_val, min_val, rms,
        kurtosis, skewness, zcr, spectral_centroid,
        peak_freq, band_energy_1_5Hz, band_energy_5_10Hz
    ]
