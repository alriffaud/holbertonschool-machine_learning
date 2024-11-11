#!/usr/bin/env python3
""" This module defines the function likelihood. """
import numpy as np


def likelihood(x, n, P):
    """
    This function calculates the likelihood of obtaining data given various
    hypothetical probabilities of developing severe side effects.
    Args:
        x (int): The number of patients that develop severe side effects.
        n (int): The total number of patients observed.
        P (np.ndarray): An array containing the various hypothetical
        probabilities of developing severe side effects.
    Returns:
        np.ndarray: An array containing the likelihood of obtaining the data,
        x and n, for each probability in P.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    error = "x must be an integer that is greater than or equal to 0"
    if not isinstance(x, int) or x < 0:
        raise ValueError(error)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate the factorials and the binomial coefficient
    n_fact = np.math.factorial(n)
    x_fact = np.math.factorial(x)
    n_x_fact = np.math.factorial(n - x)
    binom_coeff = n_fact / (x_fact * n_x_fact)

    # Calculate the likelihood of obtaining the data
    # for each probability in P
    likelihoods = binom_coeff * (P ** x) * ((1 - P) ** (n - x))

    return likelihoods
