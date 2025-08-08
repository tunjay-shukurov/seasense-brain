from ann_module.config import constants

def get_channel_list(channel_type: str):
    """
    Returns the channel list from config for the given channel type (e.g. 'BH?', 'HH?', 'LH?').

    Args:
        channel_type (str): Channel type (example: 'BH?', 'HH?', 'LH?')

    Returns:
        list[str]: List of channel codes, e.g. ['BHZ', 'BH1', 'BH2']
                   Returns an empty list if not supported.
    """
    channel_type = channel_type.upper()
    channels = constants.CHANNEL_MAP.get(channel_type, [])
    if not channels:
        print(f"⚠️ Warning: Unsupported channel type '{channel_type}'.")
    return channels
