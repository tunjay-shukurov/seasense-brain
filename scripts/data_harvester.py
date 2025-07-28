import random
from pathlib import Path
from datetime import datetime
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.clients.fdsn.header import FDSNNoDataException

# IRIS veri merkezi istemcisi
client = Client("IRIS")

# Log ve veri klasörleri
LOG_PATH = Path("ai_exports/harvester_log.txt")
RAW_SAC_BASE = Path("raw_sac_data")

def log(message: str):
    """Kısa log mesajı ekler."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{now} - {message}\n")

def log_event_block(block: str):
    """Olay bloklarını ayraçlarla log dosyasına ekler."""
    separator = "=" * 60
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"\n{separator}\n{block}\n{separator}\n\n")

def log_run_header(label: str, folder_path: Path):
    """Her çalışmanın başında özet başlık loglar."""
    header = f"""
🟢 RUN START
🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📁 Klasör: {folder_path}
----------------------------------------
""".strip()
    log_event_block(header)

def write_log_details(folder_path: Path, details: list):
    """Her eventin özetini log dosyasına tek satırda ve tekrar olmadan yazar."""
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[Klasör]: {folder_path.name}\n")
        f.write("EventID | Zaman | Koordinatlar | Magnitude | İndirilen\n")
        for ev in details:
            f.write(
                f"{ev['event_id']} | {ev['time']} | ({ev['lat']}, {ev['lon']}) | {ev['mag']} | {ev['sac_count']}\n"
            )
        f.write("\n" + "=" * 60 + "\n\n")

def ensure_base_dirs():
    """Gerekli klasörleri oluşturur."""
    RAW_SAC_BASE.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.touch(exist_ok=True)

def get_channel_list(channel_type: str):
    """Kanal tipine göre kanal listesini döndürür."""
    channel_map = {
        "HH?": ["HHZ", "HH1", "HH2"],
        "BH?": ["BHZ", "BH1", "BH2"],
        "LH?": ["LHZ", "LH1", "LH2"]
    }
    return channel_map.get(channel_type, [])

def fetch_events(starttime, endtime, minmag, maxmag):
    """Belirtilen aralık ve büyüklükteki olayları getirir."""
    try:
        events = client.get_events(
            starttime=starttime, endtime=endtime,
            minmagnitude=minmag, maxmagnitude=maxmag,
            orderby="time-asc", limit=100
        )
        return events
    except FDSNNoDataException:
        print("⚠️ Uygun olay bulunamadı.")
        return []

def get_station_triplets(inventory, channel_list):
    """İstenen kanallara sahip istasyonları döndürür."""
    triplets = []
    for net in inventory:
        for sta in net:
            chans = [cha.code for cha in sta]
            if all(k in chans for k in channel_list):
                triplets.append((net.code, sta.code))
    return triplets

def download_event_streams(event, channel_list, save_dir, target_download, downloaded_streams_total, label):
    """
    Bir olay için uygun istasyonlardan veri indirir.
    İndirilen dosya sayısı ve kullanılan istasyonlar döner.
    """
    origin = event.preferred_origin() or event.origins[0]
    magnitude = event.preferred_magnitude() or event.magnitudes[0]
    ev_time = origin.time
    mag = magnitude.mag
    timestamp = ev_time.strftime("%Y%m%d_%H%M%S")

    # İstasyonları bul
    inventory = client.get_stations(
        latitude=origin.latitude, longitude=origin.longitude,
        maxradius=20, level="channel", starttime=ev_time
    )
    station_triplets = get_station_triplets(inventory, channel_list)
    random.shuffle(station_triplets)

    downloaded_streams = 0
    station_used = []

    # Her istasyon ve kanal için veri indir
    for net, sta in station_triplets:
        if downloaded_streams_total + downloaded_streams >= target_download:
            break
        for cha in channel_list:
            if downloaded_streams_total + downloaded_streams >= target_download:
                break
            try:
                st = client.get_waveforms(
                    network=net, station=sta, location="*",
                    channel=cha, starttime=ev_time,
                    endtime=ev_time + 300
                )
                fname = f"{timestamp}_{sta}_{cha}_{label}.SAC"
                fpath = save_dir / fname
                st.write(str(fpath), format="SAC")
                downloaded_streams += 1
                station_used.append(sta)
            except Exception:
                continue

    return downloaded_streams, station_used, mag, timestamp, origin

def main():
    # Kullanıcıdan parametreleri al
    start = input("🔹 Başlangıç tarihi (YYYY-MM-DD): ")
    end = input("🔹 Bitiş tarihi (YYYY-MM-DD): ")
    minmag = float(input("🔹 Minimum büyüklük: "))
    maxmag = float(input("🔹 Maksimum büyüklük: "))
    purpose = input("🔹 Amacınız? ('train' ya da 'test'): ").strip().lower()
    target_download = int(input("🔹 Kaç tane SAC dosyası indirilecek? "))
    channel_type = input("🔹 Kanal türü (örnek: BH?, HH?, LH?): ").strip().upper()

    channel_list = get_channel_list(channel_type)
    if not channel_list:
        print(f"Desteklenmeyen kanal tipi: {channel_type}")
        return

    ensure_base_dirs()

    # Label belirle
    label = "earthquake" if minmag >= 4.5 else "noise"

    # Alt klasör ismini oluştur (ör: 20250725_153012)
    subfolder_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    if purpose == "test":
        save_dir = RAW_SAC_BASE / "test" / subfolder_name
    else:
        save_dir = RAW_SAC_BASE / label / subfolder_name

    save_dir.mkdir(parents=True, exist_ok=True)

    starttime = UTCDateTime(start)
    endtime = UTCDateTime(end)

    print("🌐 Deprem verileri aranıyor...")

    events = fetch_events(starttime, endtime, minmag, maxmag)
    if not events:
        return

    downloaded_streams_total = 0
    all_event_details = []

    log_run_header(label, save_dir)

    for event in events:
        if downloaded_streams_total >= target_download:
            break
        try:
            downloaded_streams, station_used, mag, timestamp, origin = download_event_streams(
                event, channel_list, save_dir, target_download, downloaded_streams_total, label
            )
            downloaded_streams_total += downloaded_streams

            event_block = f"""
📌 EVENT: {timestamp}
🔸 Magnitude: {mag:.1f}
🔸 Label: {label}
📍 Koordinatlar: ({origin.latitude}, {origin.longitude})
📍 Stations: {', '.join(set(station_used)) or 'None'}
📥 İndirilen: {downloaded_streams}
""".strip()
            log_event_block(event_block)
            print(event_block)

            all_event_details.append({
                "event_id": event.resource_id.id.split('/')[-1],
                "time": (origin.time).strftime('%Y-%m-%d %H:%M:%S'),
                "lat": origin.latitude,
                "lon": origin.longitude,
                "mag": mag,
                "sac_count": downloaded_streams
            })

        except Exception as e:
            print(f"⚠️ Olay atlandı: {e}")
            continue

    # Fazla dosyaları sil ve logla
    all_sac_files = sorted(save_dir.glob("*.SAC"))
    if len(all_sac_files) > target_download:
        excess = all_sac_files[target_download:]
        for f in excess:
            f.unlink()
            print(f"🗑️ Fazlalık dosya silindi: {f.name}")
        all_sac_files = sorted(save_dir.glob("*.SAC"))

    write_log_details(save_dir, all_event_details)
    log(f"✅ Run tamamlandı. Toplam indirilen: {len(all_sac_files)} dosya.\n")
    print(f"\n✅ Toplam {len(all_sac_files)} SAC dosyası başarıyla kaydedildi → {save_dir}")

if __name__ == "__main__":
    main()
