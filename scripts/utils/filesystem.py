from pathlib import Path
import numpy as np

# === Gereken klasörlerin listesi ===
REQUIRED_DIRS = [
    "data/raw_sac_data",
    "data/processed_csv",
    "logs",
    "models",
    "scripts",
]

def create_directories(dirs=REQUIRED_DIRS):
    """
    Verilen klasör listesine göre tüm klasörlerin varlığını sağlar,
    yoksa oluşturur.
    """
    for dir_path in dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"[+] Created directory: {dir_path}")
        else:
            print(f"[✓] Already exists: {dir_path}")

def ensure_dir_exists(path: Path):
    """
    Tek bir klasör yolunun varlığını kontrol eder, yoksa oluşturur.
    """
    path.mkdir(parents=True, exist_ok=True)

def get_sac_files(folder_path: Path):
    """
    Verilen klasördeki .sac uzantılı dosyaların listesini döner.
    """
    return [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() == ".sac"]

# === Prediction Log Dosyası ===
LOG_FILE_PATH = Path("logs/prediction.log")

def ensure_log_file():
    """
    Prediction log dosyasının bulunduğu klasörü oluşturur,
    log dosyasını yoksa oluşturur.
    """
    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE_PATH.touch(exist_ok=True)

# === CSV Olarak sinyal kaydet ===
def save_signal_csv(path: Path, t, signal):
    """
    Zaman ve sinyal değerlerini CSV dosyasına yazar.
    """
    np.savetxt(path, np.column_stack((t, signal)), delimiter=",", header="Time,Amplitude", comments='')

# === Eğitim Log Dosyasını hazırla ===
TRAINING_LOG_PATH = Path("logs/training_log.txt")

def ensure_training_log():
    """
    Eğitim log dosyasını oluşturur, yoksa.
    """
    TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_LOG_PATH.touch(exist_ok=True)

# === Model ve Scaler Kayıt Yardımcıları ===
def save_model(model, path: Path):
    import torch
    ensure_dir_exists(path.parent)
    torch.save(model.state_dict(), path)
    print(f"💾 Model saved to: {path}")

def save_latest_model(model, model_dir: Path):
    import torch
    path = model_dir / "latest_model.pt"
    torch.save(model.state_dict(), path)
    print(f"🗂️ Latest model updated: {path}")

def save_scaler(scaler, path: Path):
    import joblib
    ensure_dir_exists(path.parent)
    joblib.dump(scaler, path)
    print(f"📊 Scaler saved to: {path}")

if __name__ == "__main__":
    create_directories()
