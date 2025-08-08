from pathlib import Path

def create_folder_structure(base_path: Path):
    folders = [
        "models",
        "logs",
        "data/raw/earthquake",
        "data/raw/noise",
        "data/synthetic",
        "data/test",
        "data/processed",
        "outputs/reports",
        "outputs/graphs"
    ]
    for folder in folders:
        path = base_path / folder
        path.mkdir(parents=True, exist_ok=True)
        (path / ".gitkeep").touch()  # Klasör Git'e alınabilsin diye
        print(f"[+] Created folder: {path}")

def main():
    base_path = Path(__file__).resolve().parent.parent
    print(f"[i] Initializing minimal project structure in: {base_path}")
    create_folder_structure(base_path)
    print("\n[✓] Folder structure initialized. .py files will be created manually.")

if __name__ == "__main__":
    main()
