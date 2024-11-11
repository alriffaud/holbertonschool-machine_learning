#!/usr/bin/env python3
""" This module defines the intersection function. """
from scipy import special


def posterior(x, n, p1, p2):
    """
    Calculates the posterior probability that the probability of developing
    severe side effects falls within a specific range given the data.

    Parameters:
    - x: Number of patients that develop severe side effects.
    - n: Total number of patients observed.
    - p1: Lower bound on the range.
    - p2: Upper bound on the range.

    Returns:
    - The posterior probability that p is within the range [p1, p2] given x and
        n.
    """
    # Validaciones de entrada
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    error = "x must be an integer that is greater than or equal to 0"
    if not isinstance(x, int) or x < 0:
        raise ValueError(error)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float) or not (0 <= p1 <= 1):
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or not (0 <= p2 <= 1):
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    # Parameters for the Beta distribution
    alpha = x + 1
    beta = n - x + 1

    # Calculation of the cumulative probability in the interval [p1, p2]
    cdf_p2 = special.betainc(alpha, beta, p2)
    cdf_p1 = special.betainc(alpha, beta, p1)

    return cdf_p2 - cdf_p1
