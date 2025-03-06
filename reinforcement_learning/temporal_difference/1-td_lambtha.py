#!/usr/bin/env python3
"""
TD(λ) Value Estimation for Reinforcement Learning.
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    This function performs the TD(λ) algorithm to estimate the value function.
    Args:
    - env: The environment instance.
    - V: numpy.ndarray of shape (s,) containing the value estimate.
    - policy: Function that takes a state and returns the next action.
    - lambtha: The eligibility trace factor.
    - episodes: Number of episodes to train over.
    - max_steps: Maximum number of steps per episode.
    - alpha: Learning rate.
    - gamma: Discount rate.
    Returns:
    - Updated V, the updated value estimate.
    """
    # Iterate over each episode
    for _ in range(episodes):
        # Initialize eligibility traces for all states to 0
        E = np.zeros_like(V)
        state = env.reset()[0]  # Reset environment and get initial state

        # Iterate over steps within the episode
        for _ in range(max_steps):
            # Choose an action based on the policy given the current state
            action = policy(state)
            # Take a step in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Compute the TD error delta
            delta = reward + gamma * V[next_state] - V[state]

            # Update the eligibility trace for the current state
            # (accumulating traces)
            E[state] += 1

            # Update all state values with the TD error scaled by their
            # eligibility traces
            V += alpha * delta * E

            # Decay eligibility traces for all states
            E *= gamma * lambtha

            # Transition to the next state
            state = next_state

            # If the episode is terminated or truncated, break out of the loop
            if terminated or truncated:
                break
    return V
