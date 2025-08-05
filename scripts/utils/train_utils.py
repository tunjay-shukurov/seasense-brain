from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils.logger import setup_train_logger

# -------------------------------
# 1. Veriyi yükle ve hazırla
# -------------------------------
def prepare_data_from_dataframe(df, test_size=0.2, batch_size=32):
    # Özellikleri ve etiketleri ayır
    X = df.select_dtypes(include=[np.number]).values.astype(float)
    y = df["label"].values.astype(int)

    # Ölçekleme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Eğitim / Doğrulama bölmesi
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    # Torch tensörleri
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, scaler

# -------------------------------
# 2. ANN Modelini oluştur
# -------------------------------
def build_ann_model(input_dim):
    """
    Basit 3 katmanlı yapay sinir ağı modeli oluşturur.
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )
    return model

# -------------------------------
# 3. Modeli eğit
# -------------------------------
def train_model(model, train_loader, val_loader, epochs, lr, log_path, device):
    """
    Modeli verilen eğitim ve doğrulama verileriyle iteratif olarak eğitir.
    Her epoch sonunda doğrulama metrikleri hesaplanıp loglanır.
    """
    logger = setup_train_logger(log_path)
    logger.info(f"Training started on device: {device}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    logger.info("Training finished.")
    return model

# -------------------------------
# 4. Model değerlendirme
# -------------------------------
def evaluate_model(model, val_loader, criterion, device):
    """
    Modelin doğrulama veri seti üzerindeki kaybını ve doğruluk oranını hesaplar.
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / total
    return avg_val_loss, accuracy
