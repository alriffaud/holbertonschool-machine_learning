#!/usr/bin/env python3
""" This module defines the Poisson class. """


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
