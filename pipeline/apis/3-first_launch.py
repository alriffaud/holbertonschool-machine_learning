#!/usr/bin/env python3
"""
This script displays the first upcoming SpaceX launch details using the
unofficial SpaceX API.
"""
import requests


def first_launch():
    """
    Retrieves details of the first upcoming SpaceX launch based on the
    date_unix field.
    The output format is: <launch name> (<date_local>) <rocket name> -
    <launchpad name> (<launchpad locality>)
    Returns:
        str: A formatted string containing the launch name, local date,
            rocket name, launchpad name, and launchpad locality.
    """
    # Get upcoming launches from the SpaceX API
    launches_url = "https://api.spacexdata.com/v4/launches/upcoming"
    response = requests.get(launches_url)
    if response.status_code != 200:
        return None  # If request fails, return None

    launches = response.json()
    if not launches:
        return None  # No upcoming launches found

    # Sort the launches by date_unix in ascending order
    sorted_launches = sorted(launches, key=lambda x: x.get("date_unix", 0))
    # Select the first launch (the one with the earliest date_unix)
    launch = sorted_launches[0]

    # Extract launch name and local date from the launch data
    launch_name = launch.get("name")
    launch_date_local = launch.get("date_local")

    # Retrieve rocket details using the rocket ID from the launch
    rocket_id = launch.get("rocket")
    rocket_url = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    rocket_response = requests.get(rocket_url)
    if rocket_response.status_code != 200:
        rocket_name = "Unknown"
    else:
        rocket_data = rocket_response.json()
        rocket_name = rocket_data.get("name")

    # Retrieve launchpad details using the launchpad ID from the launch
    launchpad_id = launch.get("launchpad")
    launchpad_url = f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
    launchpad_response = requests.get(launchpad_url)
    if launchpad_response.status_code != 200:
        launchpad_name = "Unknown"
        launchpad_locality = "Unknown"
    else:
        launchpad_data = launchpad_response.json()
        launchpad_name = launchpad_data.get("name")
        launchpad_locality = launchpad_data.get("locality")

    # Format the output string according to the specified format
    result = f"{launch_name} ({launch_date_local}) {rocket_name} \
- {launchpad_name} ({launchpad_locality})"
    return result


if __name__ == '__main__':
    # Only execute the following when the script is run directly
    launch_info = first_launch()
    if launch_info:
        print(launch_info)
