from pathlib import Path

# =========================
# 🔧 HARDCODED PATH CONSTANTS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent  # Project root directory

LOG_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
TEST_DIR = DATA_DIR / "test"

RAW_EARTHQUAKE_DIR = RAW_DIR / "earthquake"
RAW_NOISE_DIR = RAW_DIR / "noise"

OUTPUTS_DIR = BASE_DIR / "outputs"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# =========================
# 📥 HARVESTER CONSTANTS
# =========================
IRIS_CLIENT = "IRIS"
DEFAULT_MAX_RADIUS = 20              # km
WAVEFORM_DURATION = 300              # seconds (each data segment is 5 minutes)
CHANNEL_MAP = {
    "HH?": ["HHZ", "HH1", "HH2"],
    "BH?": ["BHZ", "BH1", "BH2"],
    "LH?": ["LHZ", "LH1", "LH2"]
}
LOG_FILE = LOG_DIR / "harvester_log.txt"
DEFAULT_SAC_BASE = RAW_DIR

# 🌍 MAGS: Magnitude limits
MIN_MAGNITUDE_LIMIT = 0.0            # Minimum allowed magnitude
MAX_MAGNITUDE_LIMIT = 10.0           # Maximum allowed magnitude
MAG_THRESHOLD_EARTHQUAKE = 4.5       # Above 4.5 is "earthquake", below is "noise"

# =========================
# 🧪 SIMULATOR CONSTANTS
# =========================
SYNTHETIC_SAMPLE_COUNT = 300         # Total synthetic samples
SYNTHETIC_EARTHQUAKE_RATIO = 0.5     # 150 earthquake, 150 noise

# =========================
# 📊 GENERAL CONVENTIONS
# =========================
LABEL_EARTHQUAKE = "earthquake"
LABEL_NOISE = "noise"
PURPOSE_TRAIN = "train"
PURPOSE_TEST = "test"
SUPPORTED_PURPOSES = [PURPOSE_TRAIN, PURPOSE_TEST]

SUPPORTED_CHANNEL_TYPES = list(CHANNEL_MAP.keys())
