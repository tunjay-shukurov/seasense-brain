import logging
from pathlib import Path
from datetime import datetime
from utils.filesystem import create_directories

# Gerekli klasörleri oluştur (örneğin logs)
create_directories()

# Genel log dosyası (basit kullanım için)
GENERAL_LOG_PATH = Path("logs/harvester_log.txt")
GENERAL_LOG_PATH.touch(exist_ok=True)

# Eğitim log dosyası (train_utils vs için)
TRAINING_LOG_PATH = Path("logs/training_log.txt")
TRAINING_LOG_PATH.touch(exist_ok=True)

def log(message: str):
    """Basit zaman damgalı genel log yaz."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with GENERAL_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{now} - {message}\n")

def log_event_block(block: str):
    """Blok halinde log yazmak için."""
    separator = "=" * 60
    with GENERAL_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"\n{separator}\n{block}\n{separator}\n\n")

def log_run_header(label: str, folder_path: Path):
    """Özel run başlangıcı header logu."""
    header = f"""
🟢 RUN START
🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📁 Folder: {folder_path}
----------------------------------------
""".strip()
    log_event_block(header)

def write_log_details(folder_path: Path, details: list):
    """Event detaylarını yaz."""
    with GENERAL_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[Folder]: {folder_path.name}\n")
        f.write("EventID | Time | Coordinates | Magnitude | Downloaded\n")
        for ev in details:
            f.write(
                f"{ev['event_id']} | {ev['time']} | ({ev['lat']}, {ev['lon']}) | {ev['mag']} | {ev['sac_count']}\n"
            )
        f.write("\n" + "=" * 60 + "\n\n")

def log_predictions(predictions):
    """Tahmin sonuçlarını logla."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n=== Prediction Run at {now_str} ==="
    with GENERAL_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(header + "\n")
        for pred in predictions:
            line = f"{pred['filename']} | {pred['predicted_label']} | Confidence: {pred['confidence']}\n"
            f.write(line)
        f.write("="*40 + "\n")

def setup_train_logger(log_path: Path = TRAINING_LOG_PATH):
    """
    Eğitim sırasında kullanılacak logging.Logger nesnesini hazırlar.
    log_path: Log dosyasının tam yolu.
    """
    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)

    # Aynı logger tekrar tekrar handler eklemesin
    if not logger.hasHandlers():
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
