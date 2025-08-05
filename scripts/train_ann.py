import torch
from pathlib import Path
from datetime import datetime
from utils.logger import setup_train_logger
from utils.filesystem import create_directories
from utils.train_utils import prepare_data_from_dataframe, build_ann_model, train_model
import pandas as pd
import numpy as np

def main():
    # 1. Gerekli klasörleri oluştur
    create_directories()

    # 2. Log dosyası ve logger ayarı
    log_path = Path("logs/training_log.txt")
    logger = setup_train_logger(log_path)

    # 3. Gerçek ve sentetik verileri yükle
    real_csv = Path("data/processed_csv/real_features_extracted.csv")
    synth_csv = Path("data/processed_csv/simulated_features.csv")

    df_real = pd.read_csv(real_csv)
    df_synth = pd.read_csv(synth_csv)

    # Label sütunu var mı kontrol et
    if "label" not in df_real.columns:
        raise ValueError("real_features_extracted.csv dosyasında 'label' sütunu yok!")
    if "label" not in df_synth.columns:
        raise ValueError("simulated_features.csv dosyasında 'label' sütunu yok!")

    # Veri setlerini birleştir
    df_combined = pd.concat([df_real, df_synth], ignore_index=True)

    # 4. Veriyi hazırla (DataFrame doğrudan gönderiliyor)
    train_loader, val_loader, scaler = prepare_data_from_dataframe(df_combined, test_size=0.2, batch_size=32)
    
    # Geçici birleştirilmiş CSV dosyasına kaydet
    temp_csv_path = "data/processed_csv/temp_combined.csv"
    df_combined.to_csv(temp_csv_path, index=False)
    
    # 5. Modeli oluştur
    input_dim = next(iter(train_loader))[0].shape[1]  # Özellik sayısı
    model = build_ann_model(input_dim)

    # 6. Eğitim cihazını belirle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 7. Modeli eğit
    trained_model = train_model(model, train_loader, val_loader, epochs=30, lr=0.001, log_path=log_path, device=device)

    # 8. Modeli kaydet (zaman damgası ile ve latest_model.pt olarak)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"ann_model_{timestamp}.pt"
    model_path = models_dir / model_filename
    latest_model_path = models_dir / "latest_model.pt"

    torch.save(trained_model.state_dict(), model_path)
    torch.save(trained_model.state_dict(), latest_model_path)
    logger.info(f"Model {model_path} ve {latest_model_path} konumlarına kaydedildi.")

    logger.info("Eğitim süreci başarıyla tamamlandı.")

if __name__ == "__main__":
    main()
