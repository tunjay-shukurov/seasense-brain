import os

def rename_sac_files(sac_dir, event_date, event_mag, event_loc, label="earthquake"):
    """
    SAC dosyalarını yeniden adlandırır.

    Parametreler:
    - sac_dir: SAC dosyalarının bulunduğu klasör
    - event_date: Olay tarihi (YYYYMMDD)
    - event_mag: Büyüklük (örn. "M4.7")
    - event_loc: Bölge adı (örn. "southofkermadecisland")
    - label: "earthquake" veya "noise"
    """
    renamed_count = 0
    skipped_count = 0
    failed_count = 0

    for filename in os.listdir(sac_dir):
        if not filename.lower().endswith(".sac"):
            continue

        try:
            # Önce . ile ayırmayı dene
            parts = filename.split(".")
            if len(parts) >= 4:
                net, sta, loc, cha = parts[:4]
            else:
                # Alternatif ayırıcı: "_" (örneğin IRIS .sac adları)
                base = os.path.basename(filename)
                tokens = base.split("_")
                sta = tokens[-3]
                cha = tokens[-2]
        except Exception as e:
            print(f"❌ Format çözülemedi: {filename}")
            failed_count += 1
            continue

        new_name = f"{event_date}_{event_mag}_{event_loc}_{sta}_{cha}_{label}.SAC"
        old_path = os.path.join(sac_dir, filename)
        new_path = os.path.join(sac_dir, new_name)

        if os.path.exists(new_path):
            print(f"⚠️ Atlandı (zaten var): {new_name}")
            skipped_count += 1
            continue

        os.rename(old_path, new_path)
        print(f"✅ {filename} → {new_name}")
        renamed_count += 1

    print("\n🧾 Özet:")
    print(f"✔️ Yeniden adlandırıldı: {renamed_count}")
    print(f"⚠️ Atlananlar (zaten vardı): {skipped_count}")
    print(f"❌ Hatalı formatlar: {failed_count}")
