#!/usr/bin/env python3
"""
This module initializes the Q-table for a FrozenLake environment.
It contains the function 'q_init' that creates and returns a Q-table
as a NumPy array of zeros with shape (n_states, n_actions).
"""
import numpy as np


def q_init(env):
    """
    This function initializes the Q-table with zeros.
    Args:
        env (gym.Env): FrozenLake environment instance.
    Returns:
        numpy.ndarray: Q-table initialized with zeros, shape
        (n_states, n_actions).
    """
    # Get the number of states from the environment's observation space.
    n_states = env.observation_space.n

    # Get the number of actions from the environment's action space.
    n_actions = env.action_space.n

    # Create the Q-table as a NumPy array of zeros with shape
    # (n_states, n_actions).
    q_table = np.zeros((n_states, n_actions))

    return q_table
