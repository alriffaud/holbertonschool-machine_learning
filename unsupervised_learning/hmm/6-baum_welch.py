#!/usr/bin/env python3
""" This module defines the baum_welch function. """
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


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    This function performs the Baum-Welch algorithm for a hidden markov model.
    Args:
        Observations (numpy.ndarray): A numpy.ndarray of shape (T,) that
            contains the index of the observation. T is the number of
            observations.
        Transition (numpy.ndarray): A 2D numpy.ndarray of shape (N, N)
            containing the transition probabilities. Transition[i, j] is the
            probability of transitioning from the hidden state i to j.
        Emission (numpy.ndarray): A numpy.ndarray of shape (N, M) containing
            the emission probability of a specific observation given a hidden
            state. N is the number of hidden states in the markov model. M is
            the number of all possible observations.
        Initial (numpy.ndarray): A numpy.ndarray of shape (N, 1) containing the
            probability of starting in a particular hidden state.
        iterations (int): The number of times the algorithm should iterate.
    Returns:
        A tuple of (Transition, Emission), where Transition is the updated
        transition matrix and Emission is the updated emission matrix.
    """
    try:
        T = Observations.shape[0]  # Number of observations
        Init = Initial.shape[0]  # number of hidden states
        E = Emission.shape[1]  # number of output states

        transition_prev = Transition.copy()
        emission_prev = Emission.copy()

        for iteration in range(iterations):
            _, F = forward(Observations, Emission, Transition, Initial)
            _, B = backward(Observations, Emission, Transition, Initial)

            # Vectorized computation of NUM
            NUM = np.zeros((Init, Init, T - 1))
            for t in range(T - 1):
                Fit = F[:, t][:, np.newaxis]  # Shape (Init, 1)
                Bjt1 = B[:, t + 1]  # Shape (Init,)
                bjt1 = Emission[:, Observations[t + 1]]  # Shape (Init,)
                aij = Transition  # Shape (Init, Init)
                NUM[:, :, t] = Fit * aij * bjt1 * Bjt1[np.newaxis, :]

            DEN = np.sum(NUM, axis=(0, 1))
            X = NUM / DEN[np.newaxis, np.newaxis, :]

            # Compute gamma
            Fit = F * B  # Shape (Init, T)
            DEN_gamma = np.sum(Fit, axis=0)
            G = Fit / DEN_gamma[np.newaxis, :]

            # Update Transition matrix
            Transition = np.sum(X, axis=2) / np.sum(
                G[:, :T - 1], axis=1, keepdims=True)

            # Update Emission matrix
            DEN_emission = np.sum(G, axis=1)
            NUM_emission = np.zeros((Init, E))
            for k in range(E):
                NUM_emission[:, k] = np.sum(G[:, Observations == k], axis=1)
            Emission = NUM_emission / DEN_emission[:, np.newaxis]

            # Early stopping condition
            if np.all(np.isclose(Transition, transition_prev)) or np.all(
                    np.isclose(Emission, emission_prev)):
                return Transition, Emission

            transition_prev = Transition.copy()
            emission_prev = Emission.copy()

        return Transition, Emission

    except Exception:
        return None, None
