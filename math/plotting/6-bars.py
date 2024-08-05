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
    plt.bar(r, apples, color='r', width=0.5, label='apples')
    plt.bar(r, bananas, bottom=apples, color='y', width=0.5, label='bananas')
    plt.bar(r, oranges, bottom=np.array(apples) + np.array(bananas),
            color='#ff8000', width=0.5, label='oranges')
    plt.bar(r, peaches, bottom=np.array(apples) + np.array(bananas) +
            np.array(oranges), color='#ffe5b4', width=0.5, label='peaches')

    plt.legend()
    plt.yticks(np.arange(0, 90, 10))

    # Adding labels
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.xticks(r, people)

    # Display the plot
    plt.tight_layout()
    plt.show()
