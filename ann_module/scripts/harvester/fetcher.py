from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.clients.fdsn.header import FDSNNoDataException
from ann_module.config.constants import IRIS_CLIENT

# IRIS client instance from config
client = Client(IRIS_CLIENT)

def fetch_events(starttime: str, endtime: str, min_magnitude: float, max_magnitude: float, max_events=100):
    """
    Fetches up to max_events earthquake events from IRIS API within the given time and magnitude range.

    Args:
        starttime (str): Start time in 'YYYY-MM-DD' format
        endtime (str): End time in 'YYYY-MM-DD' format
        min_magnitude (float): Minimum magnitude
        max_magnitude (float): Maximum magnitude
        max_events (int): Maximum number of events to return

    Returns:
        obspy.events.Catalog or None: Returns None if no data is found.
    """
    try:
        start = UTCDateTime(starttime)
        end = UTCDateTime(endtime)
        catalog = client.get_events(
            starttime=start,
            endtime=end,
            minmagnitude=min_magnitude,
            maxmagnitude=max_magnitude,
            orderby="time-asc",
            limit=max_events
        )
        if len(catalog) == 0:
            print("⚠️ Warning: No data found in this range.")
            return None
        return catalog
    except FDSNNoDataException:
        print("⚠️ Warning: No data found (FDSNNoDataException).")
        return None
    except Exception as e:
        print(f"⚠️ Error: {e}")
