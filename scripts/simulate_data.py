from pathlib import Path
import pandas as pd
from utils.signal import generate_signal, extract_features
from utils.filesystem import create_directories, save_signal_csv

def main():
    create_directories()

    sim_dir = Path("data/raw_sac_data/simulated_signals")
    export_dir = Path("data/processed_csv")

    data = []
    num_per_class = 100
    labels = ["earthquake"] * num_per_class + ["noise"] * num_per_class

    for i, label in enumerate(labels):
        t, signal, fs = generate_signal(label=label)
        filename = f"sim_{i}.csv"
        file_path = sim_dir / filename
        save_signal_csv(file_path, t, signal)

        features = extract_features(signal, fs)
        data.append(features + [filename, label])

    columns = [
        "duration", "mean", "std", "max", "min", "rms",
        "kurtosis", "skewness", "zero_crossing_rate",
        "spectral_centroid", "peak_freq", "band_energy_1_5Hz",
        "band_energy_5_10Hz", "filename", "label"
    ]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(export_dir / "simulated_features.csv", index=False)

    print("✅ 200 signals generated (100 earthquake, 100 noise) and features extracted.")
    print(f"📁 Saved file: {export_dir / 'simulated_features.csv'}")

if __name__ == "__main__":
    main()
