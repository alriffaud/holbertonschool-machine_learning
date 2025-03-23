#!/usr/bin/env python3
"""
This script retrieves the location of a GitHub user using the GitHub API.
"""
import sys
import requests
import time


def get_user_location(url):
    """
    This funtion fetches the location of a GitHub user from the provided
    API URL.
    Args:
        url (str): The API URL for the GitHub user.
    Returns:
        str: The location of the user, "Not found" if user doesn't exist,
             or "Reset in X min" if rate limit exceeded.
    """
    response = requests.get(url)

    # If user is not found (404)
    if response.status_code == 404:
        return "Not found"

    # If rate limit exceeded (403)
    if response.status_code == 403:
        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
        current_time = int(time.time())
        minutes_remaining = (reset_time - current_time) // 60
        return f"Reset in {minutes_remaining} min"

    # If request is successful (200)
    if response.status_code == 200:
        data = response.json()
        return data.get("location", "Not found")

    return "Unexpected error"


if __name__ == '__main__':
    # Check if script received the correct number of arguments
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API User URL>")
        sys.exit(1)

    # Get the user URL from command-line arguments
    user_url = sys.argv[1]
    print(get_user_location(user_url))
