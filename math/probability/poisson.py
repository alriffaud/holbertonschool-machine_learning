#!/usr/bin/env python3
""" This module defines the Poisson class. """


def factorial(k):
    """
    This method calculates the factorial of a number.
    Args:
        - k: The number to calculate the factorial of.
    Returns:
        The factorial of k.
    """
    if k == 0:
        return 1
    return k * factorial(k - 1)


class Poisson:
    """ This class represents a Poisson distribution. """
    def __init__(self, data=None, lambtha=1.):
        """
        This method initializes the Poisson class.
        Args:
            - data: A list of the data to be used to estimate the distribution.
            - lambtha: The expected number of occurences in a given time frame.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """
        This method calculates the value of the PMF for a given number of
        successes.
        Args:
            - k: The number of successes.
        Returns:
            The PMF value for k.
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        e = 2.7182818285
        return (self.lambtha ** k) * (e ** -self.lambtha) / factorial(k)
