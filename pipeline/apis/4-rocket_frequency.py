#!/usr/bin/env python3
"""
This script displays the number of launches per rocket using the unofficial
SpaceX API.
"""
import requests


def rocket_frequency():
    """
    This function retrieves the number of launches per rocket from SpaceX API
    and returns a list of formatted strings.
    The output is sorted by the number of launches in descending order and
    alphabetically if tied.
    Format: <rocket name>: <number of launches>
    Returns:
        list: A list of strings, each in the format "<rocket name>:
            <number of launches>".
    """
    # Get all launches from the SpaceX API
    launches_url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(launches_url)
    if response.status_code != 200:
        return []  # Return an empty list if the request fails

    launches = response.json()
    rocket_counts = {}  # Dictionary to count launches for each rocket ID

    # Iterate over each launch and count the occurrences of each rocket
    for launch in launches:
        rocket_id = launch.get("rocket")
        if rocket_id:
            rocket_counts[rocket_id] = rocket_counts.get(rocket_id, 0) + 1

    # Dictionary to store rocket names corresponding to rocket IDs
    rockets_info = {}

    # Retrieve rocket details for each unique rocket ID to get the rocket name
    for rocket_id in rocket_counts:
        rocket_url = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
        rocket_response = requests.get(rocket_url)
        if rocket_response.status_code == 200:
            rocket_data = rocket_response.json()
            rocket_name = rocket_data.get("name", "Unknown")
            rockets_info[rocket_id] = rocket_name
        else:
            rockets_info[rocket_id] = "Unknown"

    # Create a list of tuples (rocket_name, launch_count)
    rocket_list = [(rockets_info[rid], count) for rid, count
                   in rocket_counts.items()]

    # Sort the list by number of launches (descending) and by rocket name
    # (alphabetically) if tied
    rocket_list_sorted = sorted(rocket_list, key=lambda x: (-x[1], x[0]))

    # Format each tuple into the desired output string: "<rocket name>:
    # <number of launches>"
    result = [f"{rocket_name}: {count}" for rocket_name, count
              in rocket_list_sorted]
    return result


if __name__ == '__main__':
    # Execute the code only when the script is run directly
    frequency = rocket_frequency()
    for line in frequency:
        print(line)
