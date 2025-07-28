import torch.nn as nn
import torch
import joblib

# 📦 Model tanımı
class ANNModel(nn.Module):
    def __init__(self, input_dim):
        super(ANNModel, self).__init__()
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

# 📦 Scaler'ı yükle, input boyutunu öğren
scaler = joblib.load("models/scaler.pkl")
input_dim = scaler.mean_.shape[0]

# 📦 Modeli oluştur ve state_dict'i yükle
model = ANNModel(input_dim)
state_dict = torch.load("models/latest_model.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
