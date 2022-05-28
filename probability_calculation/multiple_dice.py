# -*- coding: utf-8 -*-
"""
Created on 5/24/2022 at 13:30

@author: David.Busse
"""
from itertools import product, combinations_with_replacement
import numpy as np
import matplotlib.pyplot as plt
import scipy.special


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


def keep_k_highest_of_n(k, n, sides):
    if k > n:
        raise ValueError('x has to be smaller than n but ' + str(k) + ' is bigger than ' + str(n))

    if not type(sides) is list:
        sides = list(sides)
    sides = np.array(sides)
    values = np.sort(np.array(list(set(sides))))
    distribution = np.zeros(values.shape)
    for idx, value in enumerate(values):
        distribution[idx] = np.count_nonzero(sides == value)
    distribution /= len(sides)

    pos_results = np.arange(np.min(values)*k, np.max(values)*k+1)
    results_count = np.zeros(pos_results.shape)
    result = np.vstack((pos_results, results_count))

    idx_perms = list(combinations_with_replacement(range(len(values)), k))

    for idx_perm in idx_perms:
        min_value = np.min(values[list(idx_perm)])
        num_dice_equals_min = 0
        cur_sum = 0
        cur_prob = 1
        num_each_value = np.zeros(len(values))
        for idx in idx_perm:
            cur_sum += values[idx]

            if values[idx] == min_value:
                num_dice_equals_min += 1
            else:
                cur_prob *= distribution[idx]
                num_each_value[idx] += 1

        prob_rest_dice = get_prob_smaller_than(n-k+num_dice_equals_min, values[min(idx_perm)], values, distribution,
                                               num_equal_dice=num_dice_equals_min)
        num_poss_keep_dice = 1
        drawn = 0
        for value_amount in num_each_value:
            num_poss_keep_dice *= scipy.special.binom(n - drawn, value_amount)
            drawn += value_amount
        total_prob = cur_prob * prob_rest_dice * num_poss_keep_dice

        result_idx = np.where(result[0, :] == cur_sum)[0][0]
        result[1, result_idx] += total_prob

    # botch_prob = get_prob_smaller_than(int(n/2)+1, 0, values, probs, also_equals=False)
    return result


def get_prob_smaller_than(dice_num, value, values, probs, also_equals=True, num_equal_dice=0):
    values = np.asarray(values)
    probs = np.asarray(probs)

    value_idx = np.where(values == value)[0][0]
    if also_equals:
        value_idx += 1
    prob_lower = np.sum(probs[:value_idx])
    prob_all_lower = prob_lower ** dice_num

    if num_equal_dice > 0:
        prob_all_lower_not_equal = get_prob_smaller_than(dice_num, value, values, probs, also_equals=False)
        prob_all_lower -= prob_all_lower_not_equal
        for i in range(1, num_equal_dice):
            prob_n_i_lower = get_prob_smaller_than(dice_num-i, value, values, probs, also_equals=False)
            prob_i_equals = probs[value_idx-1] ** i
            num_poss_i_equals = scipy.special.binom(dice_num, i)
            prob_all_lower_except_i = prob_n_i_lower * prob_i_equals * num_poss_i_equals
            prob_all_lower -= prob_all_lower_except_i
            # print(dice_num, i, prob_all_lower_except_i, prob_n_i_lower, prob_i_equals, num_poss_i_equals)

    return prob_all_lower


if __name__ == '__main__':
    my_die = [-3, -2, -1, -1, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 6]

    max_difficulty = 10
    max_base_dice = 3
    max_extra_dice = 8
    success_probabilities = np.zeros((max_base_dice, max_extra_dice+1, int(max_difficulty/2)+1))
    for base_dice in range(1, max_base_dice+1):
        print('base -', base_dice)
        for extra_dice in range(max_extra_dice):
            print('--- extra -', extra_dice)
            add_results = keep_k_highest_of_n(base_dice, base_dice + extra_dice, sides=my_die)
            for difficulty in range(0, max_difficulty+1, 2):
                success_probabilities[base_dice-1, extra_dice, int(difficulty/2)] =\
                    np.sum(add_results[1, add_results[0, :] >= difficulty])

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
            plt.plot(np.arange(max_extra_dice + 1) + base_dice,
                     success_probabilities[base_dice-1, :, int(difficulty/2)],
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
    #         add_results = n_keep_k_highest(base_dice + extra_dice, base_dice, sides=my_die)
    #         success_probabilities = np.zeros(max_difficulty+1)
    #         for difficulty in range(max_difficulty+1):
    #             success_probabilities[difficulty] = np.sum(add_results[1, add_results[0, :] >= difficulty])
    #         plt.plot(np.arange(max_difficulty+1), success_probabilities, label=str(base_dice) + '_' + str(extra_dice))
    # plt.legend()
    # plt.show()

    # add_results = keep_k_highest_of_n(1, 2, my_die)
    # print(add_results)
    # plt.figure(figsize=(12, 6))
    # plt.bar(add_results[0, :], add_results[1, :], width=0.8)
    # for i in range(1, len(add_results[0, :]) + 1):
    #     plt.text(add_results[0, i - 1], add_results[1, i - 1], round(add_results[1, i - 1]*100, 2), ha='center')
    # plt.show()
