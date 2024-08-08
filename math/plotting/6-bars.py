#!/usr/bin/env python3
"""This file defines the bars function."""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    This function plots a stacked bar graph.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    r = np.arange(len(people))
    plt.bar(r, apples, color='red', width=0.5, label='apples')
    plt.bar(r, bananas, bottom=apples, color='yellow', width=0.5,
            label='bananas')
    plt.bar(r, oranges, bottom=apples + bananas, color='#ff8000', width=0.5,
            label='oranges')
    plt.bar(r, peaches, bottom=apples + bananas + oranges, color='#ffe5b4',
            width=0.5, label='peaches')

    plt.legend()
    plt.yticks(np.arange(0, 90, 10))
    plt.xticks(r, people)
    plt.ylabel('Quantity of Fruit')
    plt.title("Number of Fruit per Person")
    plt.show()
