#!/usr/bin/env python3
""" This module defines the intersection function. """
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


def intersection(x, n, P, Pr):
    """
    This function calculates the intersection of obtaining this data with
    the various hypothetical probabilities.
    Args:
        x (int): The number of patients that develop severe side effects.
        n (int): The total number of patients observed.
        P (np.ndarray): An array containing the various hypothetical
            probabilities of developing severe side effects.
        Pr (np.ndarray): An array containing the prior beliefs of P.
    Returns:
        np.ndarray: An array containing the intersection of obtaining x and n
        with each probability in P.
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
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate the likelihood of obtaining the data
    # for each probability in P
    likelihoods = likelihood(x, n, P)

    # Calculate the intersection of the obtained data with the
    # various probabilities
    intersection = Pr * likelihoods

    return intersection


def marginal(x, n, P, Pr):
    """
    This function calculates the marginal probability of obtaining the data.
    Args:
        x (int): The number of patients that develop severe side effects.
        n (int): The total number of patients observed.
        P (np.ndarray): An array containing the various hypothetical
            probabilities of developing severe side effects.
        Pr (np.ndarray): An array containing the prior beliefs of P.
    Returns:
        float: The marginal probability of obtaining x and n.
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
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate the intersection of the obtained data with the
    # various probabilities
    inter = intersection(x, n, P, Pr)

    # Calculate the marginal probability of obtaining the data
    marginal = np.sum(inter)

    return marginal


def posterior(x, n, P, Pr):
    """
    This function calculates the posterior probability for the various
    hypothetical probabilities of developing severe side effects given the
    data.
    Args:
        x (int): The number of patients that develop severe side effects.
        n (int): The total number of patients observed.
        P (np.ndarray): An array containing the various hypothetical
            probabilities of developing severe side effects.
        Pr (np.ndarray): An array containing the prior beliefs of P.
    Returns:
        np.ndarray: An array containing the posterior probability of each
        probability in P given x and n, respectively.
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
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate the intersection of the obtained data with the
    # various probabilities
    inter = intersection(x, n, P, Pr)

    # Calculate the marginal probability of obtaining the data
    marg = marginal(x, n, P, Pr)

    # Calculate the posterior probability of each probability
    posterior = inter / marg

    return posterior
