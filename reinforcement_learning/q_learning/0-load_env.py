#!/usr/bin/env python3
"""
Module to load the FrozenLake environment using gymnasium.
This module contains a function to load the FrozenLake environment
from Gymnasium, allowing the user to specify a custom map description,
a pre-made map name, and whether the environment is slippery.
"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    This function loads the pre-made FrozenLake environment from gymnasium.
    Args:
        desc (list of lists or None): Custom description of the map.
                                      If provided, should be a list of lists
                                      representing the map layout.
        map_name (str or None): Name of the pre-made map to load.
                                For example, '4x4' or '8x8'.
        is_slippery (bool): Determines whether the ice is slippery (stochastic
            behavior).
    Returns:
        gym.Env: The FrozenLake environment with the specified parameters.
    """
    # Create the FrozenLake environment using gymnasium.make with the given
    # parameters.
    env = gym.make('FrozenLake-v1', desc=desc,
                   map_name=map_name, is_slippery=is_slippery)
    return env
