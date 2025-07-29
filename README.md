# 🌊 SeaSense-Brain

SeaSense-Brain is a modular machine learning pipeline for classifying seismic activity recorded from ocean-bottom sensors. It focuses on efficient data harvesting, preprocessing of SAC files, extraction of features, and training both ANN and CNN-based neural networks.
---
## 📚 Table of Contents
- [Project Structure](#-project-structure)
- [Key Features](#-key-features)
- [Setup Instructions](#-setup-instructions)
- [Model Details](#-model-details)
- [Data Sources](#-data-sources)
- [Logging & Monitoring](#-logging--monitoring)
- [License](#-license)
## 📁 Project Structure

Root directory: `seasense-brain/`

| Folder/File     | Description                                                  |
|----------------|--------------------------------------------------------------|
| `data/`         | Contains raw and processed seismic data (SAC, CSV)           |
| `logs/`         | All training, testing, and extraction logs                   |
| `env_pt/`       | Python virtual environment and dependency configs            |
| `models/`       | Saved PyTorch model files (`.pt`)                            |
| `scripts/`      | All core Python scripts (training, prediction, simulation)   |
| `README.md`     | Project documentation (this file)                            |

## 🔧 Script Descriptions
### `gui.py`

- Displays bandpass Butterworth filtered waveform and spectrogram of `.SAC` files using a graphical interface.

### `batch_extract_real.py`

- Converts a batch of `.SAC` files into `.CSV` files with 14 extracted feature columns for model input.

### `data_harvester.py`

- Downloads `.SAC` files using the [IRIS DMC API](https://ds.iris.edu/ds/nodes/dmc/) based on custom parameters such as `time window`, `magnitude`, `channel`, and `number` of events. Supports both training and testing data.

### `feature_extractor.py`

- Functional module used by `batch_extract_real.py` to perform low-level feature extraction.

### `predictor.py`

- Uses a trained model to make predictions on `.SAC` files located in the `test_data/` directory.

### `simulate_data.py`

- Generates 200 synthetic seismic samples to improve the model’s generalization capability.

### `train_ann.py`

- Trains an `ANN` model using both real and synthetic data. Saves logs and the `confusion_matrix.png` in the `logs/` directory after training
---

## 🧠 Key Features

    - 📥 Automated SAC downloader with IRIS integration and event-based labeling

    - 📊 Feature extraction from real or synthetic seismic data

    - 🧠 Trainable ANN models using numerical input features

    - 🌐 CNN support (planned) for waveform-based classification

    - 🧪 Flexible modular scripts, built for pipeline integration

    - 📝 Advanced logging & comment parsing for AI interpretability

    - 🔧 From script → pipeline: all logic is being migrated into functional components

---

## ⚙️ Setup Instructions

```bash
#1. Clone the repository:
git clone https://github.com/tunjay-shukurov/seasense-brain.git
cd seasense-brain
```
```bash
#2. Create a virtual environment (optional):
python -m venv env_pt
source env_pt/bin/activate  # On Linux/macOS
env_pt\Scripts\activate     # On Windows
```
```bash
#3. Install dependencies:
pip install -r requirements.txt
```
```bash
#4. Run data processing:
python scripts/data_harvester.py
```
```bash
#5. Create synthetic data:
python scripts/simulate_data.py
```
```bash
#6. Train the model:
python scripts/train_ann.py
```
```bash
#7. Predict using a trained model:
python scripts/predictor.py
```
---
## 🧠 Model Details

### Current ANN architecture:
```scss
Input (n features) → Dense(64) → ReLU → Dense(32) → ReLU → Dense(1) → Sigmoid
```
### 🔵 Planned CNN
```css
Waveform Input → Conv1D Layers → GlobalPooling → Dense → Output
```
---
## 📦 Data Sources

- Real earthquake data is pulled from the [IRIS DMC API](https://ds.iris.edu/ds/nodes/dmc/)

- Synthetic noise or earthquake signals can be generated using custom waveform simulation.
---

## 🔎 Logging & Monitoring

- All runs are logged under /logs/harvester_log.log with timestamps

- Event-level summaries are auto-generated for each session

- Future versions will include comment.py to:

    - Parse logs

    - Evaluate data quality

    - Generate insights per model run

## 📃 License

MIT License © 2025 - Tunjay Shukurov

