#!/usr/bin/env python3
"""
This module interacts with the SWAPI API to retrieve a list of starships
that can carry a given number of passengers.
"""
import requests


def availableShips(passengerCount):
    """
    This function retrieves a list of starships that can accommodate at least
    `passengerCount` passengers.
    Args:
        passengerCount (int): The minimum number of passengers the ship must
            support.
    Returns:
        list: A list of starship names that meet the passenger capacity
            requirement.
    """
    url = "https://swapi-api.alx-tools.com/api/starships/"
    ships = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            return []  # Return empty list if request fails

        data = response.json()
        for ship in data["results"]:
            # Default to "0" if missing passenger data
            passengers = ship.get("passengers", "0")

            # Ignore ships with "unknown" or "n/a" passenger data
            if passengers.lower() in ["n/a", "unknown"]:
                continue

            # Convert "passengers" string to integer (handle numbers
            # with commas)
            try:
                passenger_count = int(passengers.replace(",", ""))
            except ValueError:
                continue  # Skip ship if conversion fails

            # Add ship to the list if it meets the passenger requirement
            if passenger_count >= passengerCount:
                ships.append(ship["name"])

        url = data.get("next")  # Get the next page if available

    return ships
