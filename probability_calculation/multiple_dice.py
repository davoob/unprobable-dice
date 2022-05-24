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

    if not type(sides) is list:
        sides = list(sides)
    sides = np.array(sides)
    values = np.array(list(set(sides)))
    distribution = np.zeros(values.shape)
    for idx, value in enumerate(values):
        distribution[idx] = np.count_nonzero(sides == value)
    distribution /= len(sides)

    perms = product(sides, repeat=n)

    pos_results = np.arange(np.min(values)*x, np.max(values)*x+1)
    results_count = np.zeros(pos_results.shape)
    results = np.vstack((pos_results, results_count))

    for perm in perms:
        perm = np.sort(np.array(perm))
        result = np.sum(perm[-x:])
        result_idx = np.argmin(np.abs(pos_results - result))

        results[1, result_idx] += 1
    results[1, :] /= np.sum(results[1, :])
    return results


if __name__ == '__main__':
    my_die = [-3, -2, -2, -1, -1, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6]

    max_difficulty = 8
    max_base_dice = 2
    max_extra_dice = 4
    max_dice = 5
    success_probabilities = np.zeros((max_base_dice, max_extra_dice+1, int(max_difficulty/2)+1))
    for base_dice in range(1, max_base_dice+1):
        print('base -', base_dice)
        for extra_dice in range(max_dice-base_dice+1):
            print('--- extra -', extra_dice)
            add_results = add_x_of_n(base_dice, base_dice + extra_dice, sides=my_die)
            for difficulty in range(0, max_difficulty+1, 2):
                success_probabilities[base_dice-1, extra_dice, int(difficulty/2)] = np.sum(add_results[1, add_results[0, :] >= difficulty])

    # my_die = np.array(my_die) + 1
    # for base_dice in range(1, max_base_dice + 1):
    #     print('base -', base_dice)
    #     success_probabilities = np.zeros(max_extra_dice + 1)
    #     for extra_dice in range(max_extra_dice + 1):
    #         print('--- extra -', extra_dice)
    #         add_results = add_x_of_n(base_dice, base_dice + extra_dice, sides=my_die)
    #         success_probabilities[extra_dice] = np.sum(add_results[1, add_results[0, :] >= difficulty])
    #     plt.plot(np.arange(max_extra_dice + 1) + base_dice, success_probabilities, '--', label=str(base_dice))

    plt.figure(figsize=(6, 5))
    for difficulty in range(2, max_difficulty+1, 2):
        for base_dice in range(1, max_base_dice+1):
            plt.plot(np.arange(max_extra_dice + 1) + base_dice, success_probabilities[base_dice-1, :, int(difficulty/2)],
                     label=str(difficulty) + ': ' + str(base_dice))
    plt.legend()
    plt.show()

    # my_die = np.array(my_die) + 1
    # max_difficulty = 10
    # plt.figure(figsize=(6, 5))
    # for base_dice in range(1, 3):
    #     print('base -', base_dice)
    #     for extra_dice in range(3):
    #         print('--- extra -', extra_dice)
    #         add_results = add_x_of_n(base_dice, base_dice + extra_dice, sides=my_die)
    #         success_probabilities = np.zeros(max_difficulty+1)
    #         for difficulty in range(max_difficulty+1):
    #             success_probabilities[difficulty] = np.sum(add_results[1, add_results[0, :] >= difficulty])
    #         plt.plot(np.arange(max_difficulty+1), success_probabilities, label=str(base_dice) + '_' + str(extra_dice))
    # plt.legend()
    # plt.show()

    # add_results = add_x_of_n(2, 4, my_die)
    # plt.figure(figsize=(12, 6))
    # plt.bar(add_results[0, :], add_results[1, :], width=0.8)
    # for i in range(1, len(add_results[0, :]) + 1):
    #     plt.text(add_results[0, i - 1], add_results[1, i - 1], round(add_results[1, i - 1]*100, 2), ha='center')
    # plt.show()
