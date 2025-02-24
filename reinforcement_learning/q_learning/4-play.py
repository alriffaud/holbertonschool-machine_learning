#!/usr/bin/env python3
"""
Module for playing a FrozenLake episode using a trained Q-table.
This module contains the function 'play' that makes the trained agent play
an episode by always exploiting the Q-table. Each board state is rendered
(using render_mode="ansi") and stored in a list. Note: Ensure that the
environment is created with render_mode="ansi".
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    This function has the trained agent play an episode using the learned
    Q-table.
    Args:
        env (gym.Env): The FrozenLake environment instance.
        Q (numpy.ndarray): The Q-table containing the learned Q-values.
        max_steps (int): Maximum number of steps in the episode.
    Returns:
        tuple: (total_reward, rendered_outputs) where:
            total_reward (float): The cumulative reward obtained during the
                episode.
            rendered_outputs (list): List of strings representing the rendered
                board state at each step.
    """
    rendered_outputs = []  # List to store rendered outputs of each step
    total_reward = 0.0     # Initialize total reward

    # Reset the environment and obtain the initial state.
    # For Gymnasium, reset() returns a tuple: (observation, info)
    state, _ = env.reset()

    # Iterate over steps until max_steps is reached.
    for step in range(max_steps):
        # Render and store the initial board state.
        rendered_outputs.append(env.render())

        # Always exploit the Q-table by choosing the action with the maximum
        # Q-value.
        action = np.argmax(Q[state])

        # Take the selected action in the environment.
        # For Gymnasium, step() returns: (next_state, reward, terminated,
        # truncated, info)
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Accumulate the reward.
        total_reward += reward

        # Render the new board state and append it to the rendered outputs.
        rendered_outputs.append(env.render())

        # Transition to the next state.
        state = next_state

        # If the episode is finished (either terminated or truncated), break
        # the loop.
        if terminated or truncated:
            break

    # Ensure the final state is also rendered after the episode concludes
    rendered_outputs.append(env.render())

    return total_reward, rendered_outputs
