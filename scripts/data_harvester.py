from pathlib import Path
from datetime import datetime
from obspy import UTCDateTime
from utils.logger import log, log_event_block, log_run_header, write_log_details
from utils.filesystem import create_directories
from utils.fetcher import get_channel_list, fetch_events, download_event_streams

def main():
    create_directories()
    start = input("🔹 Start date (YYYY-MM-DD): ")
    end = input("🔹 End date (YYYY-MM-DD): ")
    minmag = float(input("🔹 Minimum magnitude: "))
    maxmag = float(input("🔹 Maximum magnitude: "))
    purpose = input("🔹 Purpose? ('train' or 'test'): ").strip().lower()
    target_download = int(input("🔹 How many SAC files to download? "))
    channel_type = input("🔹 Channel type (e.g. BH?, HH?, LH?): ").strip().upper()

    channel_list = get_channel_list(channel_type)
    if not channel_list:
        print(f"Unsupported channel type: {channel_type}")
        return

    label = "earthquake" if minmag >= 4.5 else "noise"
    subfolder_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    RAW_SAC_BASE = Path("data/raw_sac_data")
    if purpose == "test":
        save_dir = RAW_SAC_BASE / "test" / subfolder_name
    else:
        save_dir = RAW_SAC_BASE / label / subfolder_name

    save_dir.mkdir(parents=True, exist_ok=True)

    starttime = UTCDateTime(start)
    endtime = UTCDateTime(end)

    print("🌐 Searching for sysmic data...")

    events = fetch_events(starttime, endtime, minmag, maxmag)
    if not events:
        return

    downloaded_streams_total = 0
    all_event_details = []

    for event in events:
        if downloaded_streams_total >= target_download:
            break
        try:
            downloaded_streams, station_used, mag, timestamp, origin = download_event_streams(
                event, channel_list, save_dir, target_download, downloaded_streams_total, label
            )
            if downloaded_streams == 0:
                print(f"⚠️ Event {timestamp} - hiç veri indirilemedi, sonraki olaya geçiliyor.")
                continue

            downloaded_streams_total += downloaded_streams

            event_block = f"""
📌 EVENT: {timestamp}
🔸 Magnitude: {mag:.1f}
🔸 Label: {label}
📍 Coordinates: ({origin.latitude}, {origin.longitude})
📍 Stations: {', '.join(set(station_used)) or 'None'}
📍 Channel: {channel_type}
📥 Downloaded this event: {downloaded_streams} | Total downloaded: {downloaded_streams_total}
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
            print(f"⚠️ Event skipped: {e}")
            continue

    all_sac_files = sorted(save_dir.glob("*.SAC"))
    if len(all_sac_files) > target_download:
        excess = all_sac_files[target_download:]
        for f in excess:
            f.unlink()
            print(f"🗑️ Excess file deleted: {f.name}")
        all_sac_files = sorted(save_dir.glob("*.SAC"))

    write_log_details(save_dir, all_event_details)
    log(f"✅ Run completed. Total downloaded: {len(all_sac_files)} files.\n")
    print(f"\n✅ Total {len(all_sac_files)} SAC files successfully saved → {save_dir}")

if __name__ == "__main__":
    main()
