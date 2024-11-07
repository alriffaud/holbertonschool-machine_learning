#!/usr/bin/env python3
""" This module contains the Normal class. """


class Normal:
    """ This class represents a normal distribution """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Constructor for the Normal class.
        Args:
            data (List): Is a list of the data to be used to estimate the
                distribution. Defaults to None.
            mean (float): Is the mean of the distribution. Defaults to 0.
            stddev (float): The standard deviation of the distribution.
                            Defaults to 1.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = (sum([(x - self.mean) ** 2 for x in data])
                           / len(data)) ** 0.5

    def z_score(self, x):
        """
        This method calculates the z-score of a given x-value.
        Args:
            x (float): The x-value.
        Returns:
            The z-score of x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        This method calculates the x-value of a given z-score.
        Args:
            z (float): The z-score.
        Returns:
            The x-value of z.
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        This method calculates the value of the PDF for a given x-value.
        Args:
            x (float): The x-value.
        Returns:
            The PDF value for x.
        """
        e = 2.7182818285
        pi = 3.1415926536
        return (1 / (self.stddev * (2 * pi) ** 0.5)
                * e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2))

    def cdf(self, x):
        """
        This method calculates the value of the CDF for a given x-value.
        Args:
            x (float): The x-value.
        Returns:
            The CDF value for x.
        """
        pi = 3.1415926536
        x = (x - self.mean) / (self.stddev * 2 ** 0.5)
        erf = (2 / pi ** 0.5
               * (x - x ** 3 / 3 + x ** 5 / 10 - x ** 7 / 42 + x ** 9 / 216))
        return (1 + erf) / 2
