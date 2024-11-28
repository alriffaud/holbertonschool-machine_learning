#!/usr/bin/env python3
""" This module defines the viterbi function. """
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """ This function performs the Viterbi algorithm for a hidden markov model.
    Args:
        Observation (numpy.ndarray): A numpy.ndarray of shape (T,) that
            contains the index of the observation. T is the number of
            observations.
        Emission (numpy.ndarray): A numpy.ndarray of shape (N, M) containing
            the emission probability of a specific observation given a hidden
            state. N is the number of hidden states in the markov model. M is
            the number of all possible observations.
        Transition (numpy.ndarray): A 2D numpy.ndarray of shape (N, N)
            containing the transition probabilities. Transition[i, j] is the
            probability of transitioning from the hidden state i to j.
        Initial (numpy.ndarray): A numpy.ndarray of shape (N, 1) containing the
            probability of starting in a particular hidden state.
    Returns:
        A tuple of (path, P), where path is a list of length T containing the
        most likely sequence of hidden states and P is the probability of
        obtaining the path sequence.
    """
    try:
        T = Observation.shape[0]
        N = Emission.shape[0]

        if Initial.shape != (N, 1) or Transition.shape != (N, N):
            return None, None

        # Initializing Viterbi and Backpointer Arrays
        V = np.zeros((N, T))
        B = np.zeros((N, T), dtype=int)

        # Initialize the first Viterbi column (t = 0)
        V[:, 0] = Initial.T * Emission[:, Observation[0]]

        # Filling the Viterbi columns (t = 1 to T-1)
        for t in range(1, T):
            for j in range(N):
                # Compute the probabilities of all possible paths
                probs = V[:, t-1] * Transition[:, j] * Emission[
                    j, Observation[t]]
                V[j, t] = np.max(probs)  # Store the maximum probability
                B[j, t] = np.argmax(probs)  # Store the backpointer

        # Reconstruct the most probable path from the backpointer array
        path = [np.argmax(V[:, T-1])]  # Last state
        for t in range(T-1, 0, -1):
            path.insert(0, B[path[0], t])  # Insert the state at the beginning

        # Compute the probability of the most probable path
        P = np.max(V[:, T-1])

        return path, P
    except Exception as e:
        return None, None
