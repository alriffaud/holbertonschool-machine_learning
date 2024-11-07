#!/usr/bin/env python3
""" This module contains the binomial class. """


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
