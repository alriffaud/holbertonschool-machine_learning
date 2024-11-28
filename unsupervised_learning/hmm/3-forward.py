#!/usr/bin/env python3
""" This module defines the forward function. """
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """ This function performs the forward algorithm for a hidden markov model.
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
        A tuple of (P, F), where P is the likelihood of the observations given
        the model, and F is a numpy.ndarray of shape (N, T) containing the
        forward path probabilities.
    """
    # Validation of the input data
    try:
        T = Observation.shape[0]  # Number of observations
        # Number of hidden states and possible observations
        N, M = Emission.shape

        if Transition.shape != (N, N) or Initial.shape != (N, 1):
            return None, None

        # Initialization of the forward path probabilities
        F = np.zeros((N, T))

        # Initialization of the first column of the forward path probabilities
        F[:, 0] = Initial.T * Emission[:, Observation[0]]

        # Recursion step of the forward algorithm
        for t in range(1, T):
            for i in range(N):
                F[i, t] = Emission[i, Observation[t]] * np.sum(
                    F[:, t-1] * Transition[:, i])

        # Total probability of the observations given the model
        P = np.sum(F[:, T-1])

        return P, F
    except Exception:
        return None, None
