#!/usr/bin/env python3
""" This module defines the absorbing function. """
import numpy as np


def absorbing(P):
    """ This function determines if a markov chain is absorbing.
    Args:
        P (numpy.ndarray): A square 2D numpy.ndarray of shape (n, n)
            representing the standard transition matrix. P[i, j] is the
            probability of transitioning from state i to state j. n is the
            number of states in the markov chain.
    Returns:
        True if it is absorbing, or False on failure.
    """
    if not isinstance(P, np.ndarray):
        return False
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return False
    if np.any(P < 0):
        return False
    if not np.allclose(P.sum(axis=1), 1):
        return False
    n = P.shape[0]
    # Validate that the rows of P add up to 1 (it is stochastic)
    if not np.allclose(P.sum(axis=1), 1):
        return False

    # Identify absorbing states
    absorbing_states = np.where(np.isclose(P.diagonal(), 1))[0]
    if len(absorbing_states) == 0:
        return False  # No absorbing states

    # Create an accessibility matrix to determine if all non-absorbing states
    # can reach some absorbing state
    reachability = np.linalg.matrix_power(P, n**2)
    for i in range(n):
        if i not in absorbing_states and not any(
                reachability[i, absorbing_states]):
            # Non-absorbent state that cannot reach an absorbent one
            return False

    return True
