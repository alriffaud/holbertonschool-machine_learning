#!/usr/bin/env python3
""" This module defines the BayesianOptimization class """
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    This class performs Bayesian optimization on a noiseless 1D Gaussian
    process. """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        This is the Class constructor.
        Args:
         - f is the black-box function to be optimized
         - X_init is a numpy.ndarray of shape (t, 1) representing the inputs
            already sampled with the black-box function
         - Y_init is a numpy.ndarray of shape (t, 1) representing the outputs
            of the black-box function for each input in X_init
         - t is the number of initial samples
         - bounds is a tuple of (min, max) representing the bounds of the space
            in which to look for the optimal point
         - ac_samples is the number of samples that should be analyzed during
            acquisition
         - l is the length parameter for the kernel
         - sigma_f is the standard deviation given to the output of the
            black-box function
         - xsi is the exploration-exploitation factor for acquisition
         - minimize is a bool determining whether optimization should be
            performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        _min, _max = bounds
        self.X_s = np.linspace(_min, _max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
        self.X = X_init
        self.Y = Y_init
