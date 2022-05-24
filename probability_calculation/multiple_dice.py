# -*- coding: utf-8 -*-
"""
Created on 5/24/2022 at 13:30

@author: David.Busse
"""
from itertools import product
import numpy as np
import matplotlib.pyplot as plt


def add_x_of_n(x, n, sides=(1, 2, 3, 4, 5, 6)):
    if x > n:
        raise ValueError('x has to be smaller than n but ' + str(x) + ' is bigger than ' + str(n))

    values = np.array(list(set(sides)))
    distribution = np.zeros(values.shape)
    for idx, value in enumerate(values):
        distribution[idx] = sides.count(value)
    distribution /= len(sides)

    perms = product(values, repeat=x)

    pos_results = np.arange(np.min(values)*x, np.max(values)*x+1)
    results_count = np.zeros(pos_results.shape)
    results = np.vstack((pos_results, results_count))

    for perm in perms:
        result = np.sum(perm)
        result_idx = np.argmin(np.abs(pos_results - result))

        rest_possibilities = np.power(len(values[values <= min(perm)]), n-x)
        results[1, result_idx] += rest_possibilities
    results[1, :] /= np.sum(results[1, :])
    return results


if __name__ == '__main__':
    add_results = add_x_of_n(2, 3)

    plt.figure(figsize=(12, 6))
    plt.bar(add_results[0, :], add_results[1, :], width=0.8)
    for i in range(1, len(add_results[0, :]) + 1):
        plt.text(add_results[0, i - 1], add_results[1, i - 1], round(add_results[1, i - 1]*100, 2), ha='center')
    plt.show()
