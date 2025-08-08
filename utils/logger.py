import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
GENERAL_LOG_FILE = LOG_DIR / "app.log"
HARVESTER_LOG_FILE = LOG_DIR / "harvester.log"

# Genel logger (app.log)
logger = logging.getLogger("seasense_logger")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(GENERAL_LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def get_logger():
    return logger


# ==== Harvester özel loglama için ayrı fonksiyonlar ====

def log_harvester_event(event_id: str, time: str, lat: float, lon: float, mag: float,
                        sac_count: int, channel_type: str, label: str):
    """
    Harvester için olay bazlı detaylı log satırı yazdırır.
    Örnek log:
    2025-08-08 22:30:10 | EVENT: abc123 | Time: 2025-08-08 21:00:00 | Coord: (39.5, 32.8) | Mag: 5.2 | SACs: 12 | Channel: BH? | Label: earthquake
    """
    log_line = (f"{event_id} | Time: {time} | Coord: ({lat:.4f}, {lon:.4f}) | "
                f"Mag: {mag:.1f} | SACs: {sac_count} | Channel: {channel_type} | Label: {label}")

    with open(HARVESTER_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{log_line}\n")


def log_harvester_run_start(label: str, folder_path: Path):
    header = f"""
==== HARVESTER RUN START ====
Time: {logging.Formatter.converter(logging.Formatter(), None) or "unknown"}
Folder: {folder_path}
Label: {label}
---------------------------
"""
    with open(HARVESTER_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(header)


def log_harvester_run_end(total_files: int):
    footer = f"==== HARVESTER RUN END - Total downloaded: {total_files} ====\n\n"
    with open(HARVESTER_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(footer)
