import random
from pathlib import Path
from datetime import datetime
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.clients.fdsn.header import FDSNNoDataException

# IRIS data center client
client = Client("IRIS")

# Log and data directories
LOG_PATH = Path("logs/harvester_log.txt")
RAW_SAC_BASE = Path("data/raw_sac_data")

def log(message: str):
    """Append a short log message."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{now} - {message}\n")

def log_event_block(block: str):
    """Append event blocks to the log file with separators."""
    separator = "=" * 60
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"\n{separator}\n{block}\n{separator}\n\n")

def log_run_header(label: str, folder_path: Path):
    """Log a summary header at the start of each run."""
    header = f"""
🟢 RUN START
🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📁 Folder: {folder_path}
----------------------------------------
""".strip()
    log_event_block(header)

def write_log_details(folder_path: Path, details: list):
    """Write a summary of each event to the log file, one line per event."""
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[Folder]: {folder_path.name}\n")
        f.write("EventID | Time | Coordinates | Magnitude | Downloaded\n")
        for ev in details:
            f.write(
                f"{ev['event_id']} | {ev['time']} | ({ev['lat']}, {ev['lon']}) | {ev['mag']} | {ev['sac_count']}\n"
            )
        f.write("\n" + "=" * 60 + "\n\n")

def ensure_base_dirs():
    """Create necessary directories."""
    RAW_SAC_BASE.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.touch(exist_ok=True)

def get_channel_list(channel_type: str):
    """Return channel list for the given type."""
    channel_map = {
        "HH?": ["HHZ", "HH1", "HH2"],
        "BH?": ["BHZ", "BH1", "BH2"],
        "LH?": ["LHZ", "LH1", "LH2"]
    }
    return channel_map.get(channel_type, [])

def fetch_events(starttime, endtime, minmag, maxmag):
    """Fetch events in the given time and magnitude range."""
    try:
        events = client.get_events(
            starttime=starttime, endtime=endtime,
            minmagnitude=minmag, maxmagnitude=maxmag,
            orderby="time-asc", limit=100
        )
        return events
    except FDSNNoDataException:
        print("⚠️ No suitable events found.")
        return []

def get_station_triplets(inventory, channel_list):
    """Return stations with the required channels."""
    triplets = []
    for net in inventory:
        for sta in net:
            chans = [cha.code for cha in sta]
            if all(k in chans for k in channel_list):
                triplets.append((net.code, sta.code))
    return triplets

def download_event_streams(event, channel_list, save_dir, target_download, downloaded_streams_total, label):
    """
    Download data for an event from suitable stations.
    Returns number of files downloaded and stations used.
    """
    origin = event.preferred_origin() or event.origins[0]
    magnitude = event.preferred_magnitude() or event.magnitudes[0]
    ev_time = origin.time
    mag = magnitude.mag
    timestamp = ev_time.strftime("%Y%m%d_%H%M%S")

    # Find stations
    inventory = client.get_stations(
        latitude=origin.latitude, longitude=origin.longitude,
        maxradius=20, level="channel", starttime=ev_time
    )
    station_triplets = get_station_triplets(inventory, channel_list)
    random.shuffle(station_triplets)

    downloaded_streams = 0
    station_used = []

    # Download data for each station and channel
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
    # Get parameters from user
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

    ensure_base_dirs()

    # Set label
    label = "earthquake" if minmag >= 4.5 else "noise"

    # Create subfolder name (e.g. 20250725_153012)
    subfolder_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    if purpose == "test":
        save_dir = RAW_SAC_BASE / "test" / subfolder_name
    else:
        save_dir = RAW_SAC_BASE / label / subfolder_name

    save_dir.mkdir(parents=True, exist_ok=True)

    starttime = UTCDateTime(start)
    endtime = UTCDateTime(end)

    print("🌐 Searching for earthquake data...")

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
📍 Coordinates: ({origin.latitude}, {origin.longitude})
📍 Stations: {', '.join(set(station_used)) or 'None'}
📥 Downloaded: {downloaded_streams}
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

    # Remove excess files and log
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
