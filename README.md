# SeaSense-Brain

SeaSense-Brain is a lightweight machine learning project for classifying seismic events recorded from ocean-bottom stations. It focuses on preprocessing SAC files, extracting features, and training simple neural network models.

---

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

---

## 🧠 Key Features

- Automated SAC file labeling based on event magnitude (threshold: M ≥ 4.5 = earthquake)
- Extraction of features from both real and synthetic data
- Training of feedforward artificial neural networks (ANN)
- Future extension: integration with CNN (Convolutional Neural Network)
- Switch from TensorFlow (.h5) to PyTorch (.pt) model format
- Modular code with clear structure and separation of logic

---

## 🚀 Getting Started

### 1. Clone the repository:

```bash
git clone https://github.com/tunjay-shukurov/seasense-brain.git
cd seasense-brain
```
### 2. Create a virtual environment (optional):
```bash
python -m venv env_pt
source env_pt/bin/activate  # On Linux/macOS
env_pt\Scripts\activate     # On Windows
```
### 3. Install dependencies:

```bash
pip install -r requirements.txt
```
### 4. Run data processing:
```bash
python scripts/data_harvester.py
```
### 5. Create synthetic data:
```bash
python scripts/simulate_data.py
```

### 6. Train the model:
```bash
python scripts/train_ann.py
```
### 7. Predict using a trained model:
```bash
python scripts/predictor.py
```
## 🧠 Model Details

Current ANN architecture:
```
Input (n features) → Dense(64) → ReLU → Dense(32) → ReLU → Dense(1) → Sigmoid
```
## 🔗 Dataset Source

All real SAC files are downloaded from [IRIS DMC](https://ds.iris.edu/ds/nodes/dmc/).
---

## Notes

    Log files are saved under logs/ and follow a timestamped structure

    Feature extraction is modularized in feature_extractor.py   

    Synthetic data can be generated with simulate_data.py

## 🔮 Future Plans

    Transition to fully convolutional CNN model

    Real-time prediction support

    Model performance dashboard (TensorBoard or matplotlib)

    Better error handling and progress tracking