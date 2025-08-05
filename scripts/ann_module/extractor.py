from pathlib import Path
from obspy import read
import numpy as np
from utils.feature_extractor import extract_features_from_signal

def extract_features_from_sac_file(sac_file: Path):
    try:
        st = read(str(sac_file))
        tr = st[0]
        data = tr.data.astype(np.float32)
        fs = tr.stats.sampling_rate
        return extract_features_from_signal(data, fs)
    except Exception as e:
        print(f"[ERROR] Could not read {sac_file.name}: {e}")
        return None

def extract_features_from_folder(folder_path: Path, label: str):
    sac_files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() == ".sac"]
    results = []

    for sac_file in sac_files:
        features = extract_features_from_sac_file(sac_file)
        if features:
            results.append(features + [sac_file.name, label])
    return results
