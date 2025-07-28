import numpy as np
import pandas as pd
import os

# === Klasörler ===
os.makedirs("simulated_signals", exist_ok=True)
os.makedirs("ai_exports", exist_ok=True)

# === Özellik çıkarıcı ===
def extract_features(signal, fs):
    n = len(signal)
    duration = n / fs
    mean = np.mean(signal)
    std = np.std(signal)
    max_amp = np.max(signal)
    min_amp = np.min(signal)
    rms = np.sqrt(np.mean(np.square(signal)))
    kurtosis = pd.Series(signal).kurt() if n > 3 else 0
    skewness = pd.Series(signal).skew() if n > 3 else 0
    zcr = ((signal[:-1] * signal[1:]) < 0).sum() / n

    freq = np.fft.rfftfreq(n, d=1/fs)
    Y = np.abs(np.fft.rfft(signal))
    spectral_centroid = np.sum(freq * Y) / np.sum(Y) if np.sum(Y) > 0 else 0
    peak_freq = freq[np.argmax(Y)] if len(Y) > 0 else 0

    band_energy_1_5Hz = np.sum(Y[(freq >= 1) & (freq <= 5)])
    band_energy_5_10Hz = np.sum(Y[(freq > 5) & (freq <= 10)])

    return [
        duration, mean, std, max_amp, min_amp, rms,
        kurtosis, skewness, zcr, spectral_centroid,
        peak_freq, band_energy_1_5Hz, band_energy_5_10Hz
    ]

# === Sinyal üretici ===
def generate_signal(duration=60, fs=100, label="noise"):
    t = np.linspace(0, duration, int(duration * fs))
    signal = np.random.normal(0, 0.05, len(t))  # beyaz gürültü

    if label == "earthquake":
        freq = np.random.uniform(1.0, 6.0)
        phase = np.random.uniform(0, 2 * np.pi)
        envelope = np.exp(-((t - duration / 2) ** 2) / (2 * (duration / 10)**2))
        burst = np.sin(2 * np.pi * freq * t + phase) * envelope
        signal += burst
        if np.random.rand() < 0.3:
            drift = 0.2 * np.sin(2 * np.pi * 0.05 * t)
            signal += drift

    return t, signal, fs

# === Ana kod ===
data = []
num_per_class = 100
labels = ["earthquake"] * num_per_class + ["noise"] * num_per_class

for i, label in enumerate(labels):
    t, signal, fs = generate_signal(label=label)
    filename = f"sim_{i}.csv"
    np.savetxt(f"simulated_signals/{filename}", np.column_stack((t, signal)), delimiter=",", header="Time,Amplitude", comments='')

    features = extract_features(signal, fs)
    data.append(features + [filename, label])

# === DataFrame ve CSV ===
columns = [
    "duration", "mean", "std", "max", "min", "rms",
    "kurtosis", "skewness", "zero_crossing_rate",
    "spectral_centroid", "peak_freq", "band_energy_1_5Hz",
    "band_energy_5_10Hz", "filename", "label"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("ai_exports/simulated_features.csv", index=False)

print("✅ 200 sinyal üretildi (100 earthquake, 100 noise) ve özellikler çıkarıldı.")
print("📁 Kaydedilen dosya: ai_exports/simulated_features.csv")
