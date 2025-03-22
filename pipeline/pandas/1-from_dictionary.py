#!/usr/bin/env python3
"""
This script creates a DataFrame from a dictionary
"""
import pandas as pd


# Create a dictionary with two columns
dict = {
    "First": [0.0, 0.5, 1.0, 1.5],
    "Second": pd.Categorical(["one", "two", "three", "four"]),
}

# Define the row names
rows = ["A", "B", "C", "D"]

# Create a DataFrame from the dictionary
df = pd.DataFrame(dict, index=rows)
