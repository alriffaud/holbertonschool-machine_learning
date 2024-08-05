#!/usr/bin/env python3
"""This file defines the line function."""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    This function plots y as a line graph.
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, 'r')
    plt.axis([0, 10, None, None])
    plt.show()
