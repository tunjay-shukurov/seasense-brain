from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException
from pathlib import Path
import random
from ann_module.config import constants

client = Client("IRIS")

def download_event_streams(event, channel_list, save_dir: Path, target_download: int, downloaded_so_far: int, label: str):
    """
    Downloads waveform data for a given seismic event from IRIS.

    Args:
        event: Obspy Event object (seismic event)
        channel_list: List[str] - Channels to download (e.g. ["BHZ","BH1","BH2"])
        save_dir: Path - Directory to save the data
        target_download: int - Total number of files to download
        downloaded_so_far: int - Number of files downloaded so far
        label: str - "earthquake" or "noise" label

    Returns:
        downloaded_count: int - Number of files downloaded for this event
        used_stations: list[str] - Used station codes
        magnitude: float - Event magnitude
        timestamp: str - Event time (YYYYMMDD_HHMMSS)
        origin: Origin object - Event origin info
    """
    origin = event.preferred_origin() or event.origins[0]
    magnitude = event.preferred_magnitude() or event.magnitudes[0]
    ev_time = origin.time
    mag = magnitude.mag
    timestamp = ev_time.strftime("%Y%m%d_%H%M%S")

    # Query stations around the event (max radius from constants.py)
    try:
        inventory = client.get_stations(
            latitude=origin.latitude,
            longitude=origin.longitude,
            maxradius=constants.DEFAULT_MAX_RADIUS,
            level="channel",
            starttime=ev_time,
            endtime=ev_time + constants.WAVEFORM_DURATION
        )
    except FDSNException as e:
        print(f"⚠️ Station query error: {e}")
        return 0, [], mag, timestamp, origin

    # Filter stations that have all required channels
    station_list = []
    for net in inventory:
        for sta in net:
            chans = [cha.code for cha in sta.channels]
            if all(ch in chans for ch in channel_list):
                station_list.append((net.code, sta.code))

    random.shuffle(station_list)

    downloaded_count = 0
    used_stations = []

    for net, sta in station_list:
        if downloaded_so_far + downloaded_count >= target_download:
            break
        for cha in channel_list:
            if downloaded_so_far + downloaded_count >= target_download:
                break
            try:
                st = client.get_waveforms(
                    network=net,
                    station=sta,
                    location="*",
                    channel=cha,
                    starttime=ev_time,
                    endtime=ev_time + constants.WAVEFORM_DURATION
                )
                filename = f"{timestamp}_{sta}_{cha}_{label}.SAC"
                filepath = save_dir / filename
                st.write(str(filepath), format="SAC")
                downloaded_count += 1
                used_stations.append(sta)
            except Exception as e:
                print(f"⚠️ Download error: {e}")
                continue

    return downloaded_count, used_stations, mag, timestamp, origin
