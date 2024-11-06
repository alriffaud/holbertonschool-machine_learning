#!/usr/bin/env python3
""" This module defines the Exponential class. """


class Exponential:
    """ This class represents a Exponential distribution. """
    def __init__(self, data=None, lambtha=1.):
        """
        This method initializes the Exponential class.
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
            self.lambtha = len(data) / sum(data)
    
    def pdf(self, x):
        """
        This method calculates the value of the PDF for a given time period.
        Args:
            - x: The time period.
        Returns:
            The PDF value for x.
        """
        if x < 0:
            return 0
        e = 2.7182818285
        return self.lambtha * (e ** (-self.lambtha * x))
