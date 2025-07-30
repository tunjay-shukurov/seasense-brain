from pathlib import Path

# === Define required directories ===
REQUIRED_DIRS = [
    "data/raw_sac_data",
    "data/processed_csv",
    "data/prediction_results",
    "logs",
    "models",
    "scripts",
]

def create_directories():
    for dir_path in REQUIRED_DIRS:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"[+] Created directory: {dir_path}")
        else:
            print(f"[✓] Already exists: {dir_path}")

if __name__ == "__main__":
    create_directories()
