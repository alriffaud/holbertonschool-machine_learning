#!/usr/bin/env python3
""" This module defines the markov_chain function. """
import numpy as np


def markov_chain(P, s, t=1):
    """ This function determines the probability of a markov chain being in a
        particular state after a specified number of iterations.
    Args:
        P (numpy.ndarray): A square 2D numpy.ndarray of shape (n, n)
            representing the transition matrix. P[i, j] is the probability of
            transitioning from state i to state j. n is the number of states in
            the markov chain
        s (numpy.ndarray): A 2D numpy.ndarray of shape (1, n) representing the
            initial state.
        t (int): The number of iterations that the markov chain has been
            through.
    Returns:
        A numpy.ndarray of shape (1, n) representing the probability of being
        in a specific state after t iterations, or None on failure.
    """
    if not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray):
        return None
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None
    if P.shape[0] != s.shape[1]:
        return None
    if not np.allclose(P.sum(axis=1), 1):
        return None
    if not isinstance(t, int) or t < 0:
        return None
    try:
        # Compute P^t
        P_t = np.linalg.matrix_power(P, t)
        # Compute s_t
        s_t = np.matmul(s, P_t)
        return s_t
    except Exception:
        return None
