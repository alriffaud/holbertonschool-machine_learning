#!/usr/bin/env python3
""" This module defines the regular function. """
import numpy as np


def regular(P):
    """ This function determines the steady state probabilities of a regular
        markov chain.
    Args:
        P (numpy.ndarray): A square 2D numpy.ndarray of shape (n, n)
            representing the transition matrix. P[i, j] is the probability of
            transitioning from state i to state j. n is the number of states in
            the markov chain
    Returns:
        A numpy.ndarray of shape (1, n) containing the steady state
        probabilities, or None on failure.
    """
    if not isinstance(P, np.ndarray):
        return None
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None
    if not np.allclose(P.sum(axis=1), 1):
        return None
    if np.any(P <= 0):
        return None
    if np.any(P >= 1):
        return None
    n, m = P.shape
    # Check if P is regular
    try:
        # Raise P to a sufficiently high power
        P_k = np.linalg.matrix_power(P, n**2)
        if not (P_k > 0).all():
            return None
    except Exception:
        return None

    # Solve the system of equations to find the steady state probabilities
    try:
        # Transpose and subtract the identity matrix
        A = P.T - np.eye(n)
        # Add the constraint: sum(pi) = 1
        A = np.vstack([A, np.ones((1, n))])
        b = np.zeros((n + 1,))
        b[-1] = 1

        # Solve the system of equations using least squares
        pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return pi.reshape(1, -1)
    except Exception:
        return None
