#!/usr/bin/env python3
""" This module contains the binomial class. """


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


class Binomial:
    """ This class represents a binomial distribution """
    def __init__(self, data=None, n=1, p=0.5):
        """
        Constructor for the Binomial class.
        Args:
            data (List): Is a list of the data to be used to estimate the
                distribution. Defaults to None.
            n (int): The number of Bernoulli trials. Defaults to 1.
            p (float): The probability of a "success". Defaults to 0.5.
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = round(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Caclculate mean
            mean = sum(data) / len(data)
            # Calaculate stddev
            variance = (sum([(x - mean) ** 2 for x in data])
                        / len(data))
            # calculate the lambtha value
            p = 1 - (variance / mean)
            self.n = round(mean / p)
            self.p = mean / self.n

    def pmf(self, k):
        """
        This method calculates the value of the PMF for a given number of
        successes.
        Args:
            k (int): The number of successes.
        Returns:
            The PMF value for k.
        """
        if k < 0:
            return 0
        k = round(k)
        n = self.n
        p = self.p
        q = 1 - p
        return (factorial(n) / (factorial(k) * factorial(n - k))
                * (p ** k) * (q ** (n - k)))

    def cdf(self, k):
        """
        This method calculates the value of the CDF for a given number of
        successes.
        Args:
            k (int): The number of successes.
        Returns:
            The CDF value for k.
        """
        if k < 0:
            return 0
        k = round(k)
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
