import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import pandas as pd
from utils.filesystem import create_directories
from ann_module.extractor import extract_features_from_folder

create_directories()

def main():
    root = tk.Tk()
    root.withdraw()

    eq_folder = filedialog.askdirectory(title="Select Earthquake SAC folder")
    if not eq_folder:
        messagebox.showinfo("Info", "Earthquake folder not selected, operation cancelled.")
        return

    noise_folder = filedialog.askdirectory(title="Select Noise SAC folder")
    if not noise_folder:
        messagebox.showinfo("Info", "Noise folder not selected, operation cancelled.")
        return

    eq_path = Path(eq_folder)
    noise_path = Path(noise_folder)

    eq_data = extract_features_from_folder(eq_path, "earthquake")
    noise_data = extract_features_from_folder(noise_path, "noise")

    all_data = eq_data + noise_data
    if not all_data:
        messagebox.showerror("Error", "No data could be extracted.")
        return

    columns = [
        "duration", "mean", "std", "max", "min", "rms",
        "kurtosis", "skewness", "zero_crossing_rate",
        "spectral_centroid", "peak_freq", "band_energy_1_5Hz",
        "band_energy_5_10Hz", "filename", "label"
    ]

    df = pd.DataFrame(all_data, columns=columns)

    export_dir = Path("data/processed_csv")
    export_path = export_dir / "real_features_extracted.csv"
    df.to_csv(export_path, index=False)

    messagebox.showinfo("Success", f"Feature extraction completed.\nSaved to:\n{export_path}")

if __name__ == "__main__":
    main()
