import numpy as np
import pandas as pd
import torch
import joblib
from obspy import read
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from torch import nn
from utils.logger import log_event_block 
from utils.filesystem import ensure_log_file
from datetime import datetime

# ANN model architecture
class ANNModel(nn.Module):
    def __init__(self, input_dim):
        super(ANNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Feature extraction
def extract_features_from_signal(data, fs):
    n = len(data)
    duration = n / fs
    mean_val = np.mean(data)
    std_val = np.std(data)
    max_val = np.max(data)
    min_val = np.min(data)
    rms = np.sqrt(np.mean(data**2))
    kurtosis = pd.Series(data).kurt()
    skewness = pd.Series(data).skew()
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

def log_predictions(log_path: Path, folder_path: Path, predictions: list):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"""
🟢 PREDICTION RUN START
🕒 {now_str}
📁 Folder: {folder_path}
----------------------------------------
""".strip()
    log_event_block(header)

    with log_path.open("a", encoding="utf-8") as f:
        f.write(header + "\n")
        for pred in predictions:
            line = f"{pred['filename']} | {pred['predicted_label']} | Confidence: {pred['confidence']}\n"
            f.write(line)
        f.write("="*60 + "\n\n")

# Main prediction function
def main():
    ensure_log_file()
    LOG_PATH = Path("logs/prediction.log")
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select SAC folder for prediction")
    if not folder_path:
        messagebox.showinfo("Info", "No folder selected.")
        return

    folder_path = Path(folder_path)
    sac_files = [f for f in folder_path.rglob("*") if f.is_file() and f.suffix.lower() == ".sac"]
    if not sac_files:
        messagebox.showerror("Error", "No .sac files found in the folder.")
        return

    scaler = joblib.load("models/scaler.pkl")
    input_dim = scaler.mean_.shape[0]

    model = ANNModel(input_dim)
    state_dict = torch.load("models/latest_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    predictions = []

    for file in sac_files:
        tr = read(str(file))[0]
        data = tr.data.astype(np.float32)
        fs = tr.stats.sampling_rate
        features = extract_features_from_signal(data, fs)

        X_scaled = scaler.transform([features])
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            output = model(X_tensor)
            prob = output.item()
            label = "earthquake" if prob >= 0.5 else "noise"

        predictions.append({
            "filename": file.name,
            "predicted_label": label,
            "confidence": round(prob, 4)
        })

    log_predictions(LOG_PATH, folder_path, predictions)

    messagebox.showinfo("Completed", f"Prediction completed.\nLog saved to {LOG_PATH}")
    print(f"Prediction log saved at {LOG_PATH}")
    for p in predictions:
        print(f"{p['filename']} -> {p['predicted_label']} (Conf: {p['confidence']})")

if __name__ == "__main__":
    main()
