#!/usr/bin/env python3
"""
This module implements the epsilon-greedy strategy for action selection in
Q-learning. It contains the function 'epsilon_greedy' that selects the next
action based on the given Q-table, current state, and epsilon value. It uses
a random sample to decide whether to explore or exploit.
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    This function selects the next action using the epsilon-greedy strategy.
    Args:
        Q (numpy.ndarray): The Q-table containing the estimated values for each
            state-action pair.
        state (int): The current state index.
        epsilon (float): The probability threshold for choosing exploration
            over exploitation.
    Returns:
        int: The index of the selected action.
    """
    # Sample a random probability from a uniform distribution between 0 and 1.
    p = np.random.uniform(0, 1)

    # If the random probability is less than epsilon, explore: choose a random
    # action.
    if p < epsilon:
        # Choose a random action index from all possible actions.
        next_action = np.random.randint(0, Q.shape[1])
    else:
        # Otherwise, exploit: choose the action with the maximum Q value for
        # the current state.
        next_action = np.argmax(Q[state])

    return next_action
