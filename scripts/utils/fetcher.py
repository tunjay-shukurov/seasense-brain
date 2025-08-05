import random
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

client = Client("IRIS")

def get_channel_list(channel_type: str):
    channel_map = {
        "HH?": ["HHZ", "HH1", "HH2"],
        "BH?": ["BHZ", "BH1", "BH2"],
        "LH?": ["LHZ", "LH1", "LH2"]
    }
    return channel_map.get(channel_type, [])

def fetch_events(starttime, endtime, minmag, maxmag):
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
    triplets = []
    for net in inventory:
        for sta in net:
            chans = [cha.code for cha in sta]
            if all(k in chans for k in channel_list):
                triplets.append((net.code, sta.code))
    return triplets

def download_event_streams(event, channel_list, save_dir, target_download, downloaded_streams_total, label):
    origin = event.preferred_origin() or event.origins[0]
    magnitude = event.preferred_magnitude() or event.magnitudes[0]
    ev_time = origin.time
    mag = magnitude.mag
    timestamp = ev_time.strftime("%Y%m%d_%H%M%S")

    inventory = client.get_stations(
        latitude=origin.latitude, longitude=origin.longitude,
        maxradius=20, level="channel", starttime=ev_time
    )
    station_triplets = get_station_triplets(inventory, channel_list)
    random.shuffle(station_triplets)

    downloaded_streams = 0
    station_used = []

    try:
        st = client.get_waveforms(...)
        fpath = save_dir / fname
        st.write(str(fpath), format="SAC")
        downloaded_streams += 1
        station_used.append(sta)
        print(f"✔️ İndirildi: {fname}")  
    except Exception as ex:
        print(f"❌ İndirme hatası: {sta} {cha} - {ex}")


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
