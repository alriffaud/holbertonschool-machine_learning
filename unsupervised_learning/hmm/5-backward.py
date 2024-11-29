#!/usr/bin/env python3
""" This module defines the backward function. """
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    This function performs the backward algorithm for a hidden markov model.
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
        A tuple of (P, B), where P is the likelihood of the observations given
        the model and B is a numpy.ndarray of shape (N, T) containing the
        backward path probabilities.
    """
    try:
        T = Observation.shape[0]  # Number of observations
        N = Transition.shape[0]  # Number of hidden states

        # Initialize matrix B
        B = np.zeros((N, T))

        # Base step: fill the last column with 1
        B[:, T - 1] = 1

        # Recursive step: calculate the probabilities backwards
        # From the penultimate step to the beginning
        for t in range(T - 2, -1, -1):
            for i in range(N):
                # Add up the contributions of each future state
                B[i, t] = np.sum(
                    B[:, t + 1] * Transition[i, :]
                    * Emission[:, Observation[t + 1]]
                )

        # Calculate the total probability of the observed sequence
        P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])

        return P, B

    except Exception:
        return None, None
