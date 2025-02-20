#!/usr/bin/env python3
"""
Module for Q-learning training algorithm.
This module contains the function 'train' that trains a FrozenLake agent using
the Q-learning algorithm with an epsilon-greedy exploration strategy.
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    This function trains a Q-learning agent on the given environment.
    Args:
        env (gym.Env): The FrozenLake environment instance.
        Q (numpy.ndarray): The Q-table to be updated.
        episodes (int): Total number of training episodes.
        max_steps (int): Maximum steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Initial epsilon value for the epsilon-greedy policy.
        min_epsilon (float): Minimum value to which epsilon can decay.
        epsilon_decay (float): Decay rate for epsilon after each episode.
    Returns:
        tuple: (Q, total_rewards) where
            Q (numpy.ndarray): The updated Q-table.
            total_rewards (list): List containing the total reward per episode.
    """
    total_rewards = []  # List to store total reward per episode

    # Get the number of columns from the environment's description.
    # This is used to determine the row and column for a given state index.
    n_cols = len(env.unwrapped.desc[0])

    # Iterate over episodes
    for episode in range(episodes):
        # Reset the environment and obtain the initial state.
        # For Gymnasium, reset() returns a tuple: (observation, info)
        state, _ = env.reset()
        episode_reward = 0  # Initialize the reward for this episode

        # Iterate over steps within the episode
        for step in range(max_steps):
            # Select the next action using the epsilon-greedy strategy.
            action = epsilon_greedy(Q, state, epsilon)

            # Take the selected action in the environment.
            # For Gymnasium, step() returns: (next_state, reward, terminated,
            # truncated, info)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Check if the agent falls in a hole.
            # Calculate the row and column from the state index.
            row = next_state // n_cols
            col = next_state % n_cols
            if env.unwrapped.desc[row][col] == b'H':
                reward = -1  # Update reward to -1 if the agent falls in a hole

            # Q-learning update rule:
            # Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(next_state))
            # - Q(s, a))
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action])

            # Accumulate the reward for the episode
            episode_reward += reward

            # Transition to the next state
            state = next_state

            # If the episode is terminated or truncated (agent reaches goal or
            # falls in a hole), break out of the loop.
            if terminated or truncated:
                break

        # Append the total reward obtained in this episode to the rewards list.
        total_rewards.append(episode_reward)

        # Decay epsilon after each episode, ensuring it does not fall below
        # min_epsilon.
        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    return Q, total_rewards
