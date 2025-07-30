import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from datetime import datetime

# === Directories ===
data_dir = Path("data")
export_dir = data_dir / "processed_csv"
log_dir = Path("logs")
model_dir = Path("models")

# === File paths ===
real_features_path = export_dir / "real_features_extracted.csv"
simulated_features_path = export_dir / "simulated_features.csv"
conf_matrix_path = log_dir / "conf_matrix.png"
log_file = log_dir / "training_log.txt"

# === Logging Function ===
def log_training_results(model_name, loss, accuracy, report_text, cm=None, cm_path=None, dataset_info="real+simulated"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write("="*60 + "\n")
        f.write(f"[{timestamp}] MODEL: {model_name}\n")
        f.write(f"Dataset: {dataset_info}\n")
        f.write(f"Loss: {loss:.4f} | Accuracy: {accuracy*100:.2f}%\n")
        f.write("\n--- Classification Report ---\n")
        f.write(report_text)
        if cm is not None:
            f.write("\n--- Confusion Matrix (array) ---\n")
            f.write(np.array2string(cm))
        if cm_path:
            f.write(f"\nConfusion Matrix Image: {cm_path}")
        f.write("\n\n")

# === Evaluation ===
def evaluate_model(model, X_test, y_test, label_encoder, device, save_path=None):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
        outputs = model(inputs).cpu().numpy()
    
    num_classes = len(np.unique(y_test))
    if num_classes == 2:
        y_pred = (outputs > 0.5).astype(int).flatten()
    else:
        y_pred = np.argmax(outputs, axis=1)

    labels = np.unique(y_test)
    target_names = label_encoder.inverse_transform(labels)

    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Confusion matrix saved to: {save_path}")
    plt.close()

    return report, cm

# === Data Preparation ===
real_df = pd.read_csv(real_features_path)
sim_df = pd.read_csv(simulated_features_path)
combined_df = pd.concat([real_df, sim_df], ignore_index=True)

le = LabelEncoder()
combined_df["label_enc"] = le.fit_transform(combined_df["label"])

X = combined_df.drop(columns=["filename", "label", "label_enc"]).values
y = combined_df["label_enc"].values

print(f"Feature shape: {X.shape}")

# === Standard Scaler ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, model_dir / "scaler.pkl")

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === PyTorch Dataset and DataLoader ===
class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = FeatureDataset(X_train, y_train)
test_dataset = FeatureDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# === Model Definition ===
class ANN(nn.Module):
    def __init__(self, input_dim):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ANN(X_train.shape[1]).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Training Loop ===
EPOCHS = 30

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device).unsqueeze(1)
            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(test_loader)
    val_acc = correct / total

    print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

# === Save Model ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
acc_percent = int(val_acc * 100)
model_name = f"sac_classifier_{timestamp}_acc{acc_percent}.pt"
model_path = model_dir / model_name
torch.save(model.state_dict(), model_path)
print(f"💾 Model saved: {model_path}")

# Update latest_model.pt
latest_model_path = model_dir / "latest_model.pt"
torch.save(model.state_dict(), latest_model_path)
print(f"🗂️ Latest model updated: {latest_model_path}")

# === Evaluation and Logging ===
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor).cpu().numpy()

report_text, cm = evaluate_model(model, X_test, y_test, le, device, save_path=conf_matrix_path)

log_training_results(
    model_name=model_name,
    loss=avg_val_loss,
    accuracy=val_acc,
    report_text=report_text,
    cm=cm,
    cm_path=conf_matrix_path,
    dataset_info="real+simulated"
)

print("✅ Training completed and results logged.")
