def get_user_params():
    """
    Gets user input and performs validation.
    This function is called at the start of the pipeline.
    """

    # --- Date range ---
    while True:
        start_date = input("Start date (YYYY-MM-DD): ").strip()
        end_date = input("End date (YYYY-MM-DD): ").strip()
        if len(start_date) == 10 and len(end_date) == 10:
            break
        print("Invalid date format, please try again.")

    # --- Magnitude range ---
    while True:
        try:
            min_magnitude = float(input("Minimum magnitude (e.g. 2.5): ").strip())
            max_magnitude = float(input("Maximum magnitude (e.g. 6.0): ").strip())
            if min_magnitude <= max_magnitude:
                break
            else:
                print("Minimum magnitude must be less than or equal to maximum.")
        except ValueError:
            print("Please enter a valid number.")

    # --- Channel type ---
    valid_channels = ["HH?", "BH?", "LH?"]
    while True:
        channel_type = input(f"Channel type ({', '.join(valid_channels)}): ").strip().upper()
        if channel_type in valid_channels:
            break
        print(f"Invalid channel type. Valid options: {', '.join(valid_channels)}")

    # --- Data purpose ---
    valid_purposes = ["train", "test"]
    while True:
        purpose = input(f"Data purpose ({', '.join(valid_purposes)}): ").strip().lower()
        if purpose in valid_purposes:
            break
        print(f"Invalid data purpose. Options: {', '.join(valid_purposes)}")

    # --- Network ---
    valid_networks = ["IU", "II", "GE", "AZ", "AV", "US", "UW", "*"]
    network_help = (
        "Example network codes:\n"
        "  IU - Global Seismic Network (GSN) / IRIS/USGS\n"
        "  II - IRIS/IDA\n"
        "  GE - GEOFON\n"
        "  AZ, AV, US, UW - Regional networks\n"
        "  * - All networks (default)\n"
    )
    print(network_help)
    while True:
        network = input(f"Network code ({', '.join(valid_networks)}): ").strip().upper()
        if network in valid_networks:
            break
        print(f"Invalid network code. Valid options: {', '.join(valid_networks)}")

    # --- Number of SAC files to download ---
    while True:
        try:
            sac_count = int(input("Number of SAC files to download: ").strip())
            if sac_count > 0:
                break
            else:
                print("Number must be positive.")
        except ValueError:
            print("Please enter a valid integer.")

    return {
        "start_date": start_date,
        "end_date": end_date,
        "min_magnitude": min_magnitude,
        "max_magnitude": max_magnitude,
        "channel_type": channel_type,
        "purpose": purpose,
        "network": network,
        "sac_count": sac_count
    }

if __name__ == "__main__":
    params = get_user_params()
    print("\nGirilen parametreler:")
    for k, v in params.items():
        print(f"{k}: {v}")
