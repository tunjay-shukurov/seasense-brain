from obspy import read
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

# === Feature extraction function ===
def extract_features_from_signal(data, fs):
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

def extract_from_folder(folder_path: Path, label: str):
    sac_files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() == ".sac"]
    if not sac_files:
        messagebox.showerror("Error", f"No .sac files found in the {label} folder.")
        return []

    data_rows = []
    for sac_file in sac_files:
        try:
            st = read(str(sac_file))
            tr = st[0]
            data = tr.data.astype(np.float32)
            fs = tr.stats.sampling_rate
            features = extract_features_from_signal(data, fs)
            data_rows.append(features + [sac_file.name, label])
        except Exception as e:
            print(f"ERROR: Could not read {sac_file.name} -> {e}")
    return data_rows

def main():
    root = tk.Tk()
    root.withdraw()

    # Select Earthquake folder
    eq_folder = filedialog.askdirectory(title="Select Earthquake SAC folder")
    if not eq_folder:
        messagebox.showinfo("Info", "Earthquake folder not selected, operation cancelled.")
        return

    # Select Noise folder
    noise_folder = filedialog.askdirectory(title="Select Noise SAC folder")
    if not noise_folder:
        messagebox.showinfo("Info", "Noise folder not selected, operation cancelled.")
        return

    eq_path = Path(eq_folder)
    noise_path = Path(noise_folder)

    # Feature extraction
    eq_data = extract_from_folder(eq_path, "earthquake")
    noise_data = extract_from_folder(noise_path, "noise")

    all_data = eq_data + noise_data
    if not all_data:
        messagebox.showerror("Error", "No data could be extracted from any file.")
        return

    columns = [
        "duration", "mean", "std", "max", "min", "rms",
        "kurtosis", "skewness", "zero_crossing_rate",
        "spectral_centroid", "peak_freq", "band_energy_1_5Hz",
        "band_energy_5_10Hz", "filename", "label"
    ]

    df = pd.DataFrame(all_data, columns=columns)

    export_dir = Path("data")
    export_dir.mkdir(exist_ok=True)
    export_path = export_dir / "processed_csv/real_features_extracted.csv"
    df.to_csv(export_path, index=False)

    messagebox.showinfo("Success", f"Feature extraction completed.\nSaved file:\n{export_path}")

if __name__ == "__main__":
    main()
