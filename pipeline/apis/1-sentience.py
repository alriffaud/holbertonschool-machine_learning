#!/usr/bin/env python3
"""
This module retrieves the home planets of all sentient species
from the Star Wars API (SWAPI).
"""
import requests


def sentientPlanets():
    """
    This funtion retrieves a list of names of the home planets of all sentient
    species from the SWAPI (Star Wars API).
    Returns:
        list: A list of planet names where sentient species originate.
    """
    url = "https://swapi-api.alx-tools.com/api/species/"
    planets = set()  # Use a set to avoid duplicate planet names

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            return []  # Return an empty list if request fails

        data = response.json()

        for species in data["results"]:
            # Check if species is sentient
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()

            if "sentient" in classification or "sentient" in designation:
                homeworld_url = species.get("homeworld")

                if homeworld_url:
                    # Fetch the planet name using its API URL
                    planet_response = requests.get(homeworld_url)
                    if planet_response.status_code == 200:
                        planet_name = planet_response.json().get("name")
                        if planet_name:
                            planets.add(planet_name)

        url = data.get("next")  # Get the next page if available

    return list(planets)
