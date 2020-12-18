import numpy as np
import scipy.special
import itertools
import matplotlib.pyplot as plt

class Roll:
    def __init__(self, dice):
        self.dice = dice

    def possible_outcomes(self):
        pass


def sum_same_dice_outcomes(dice):
    # assumes equal dice with uniform 1 step between values
    num_dice = len(dice)
    min_value = min(dice[0])
    max_value = max(dice[0])

    outcomes = [i for i in range(min_value*num_dice, max_value*num_dice+1)]
    return outcomes


def get_poss_num_for_sum(n, k, exclude_zero=True):
    '''
    get number of all possibilities to form the sum to n out of k numbers
    :param n: value of the sum
    :param k: amount of numbers to be summed
    :param exclude_zero: bool to exclude zero from the possible summands
    :return: number off possibilities to form the sum
    '''

    if exclude_zero:
        possibilities = np.math.factorial(n-1) / (np.math.factorial(k-1) * np.math.factorial(n-k))
    else:
        possibilities = np.math.factorial(n+k-1) / (np.math.factorial(k-1) * np.math.factorial(n))
    return possibilities


def n_keep_k_highest(n, k, values, probs):
    values = np.asarray(values)
    probs = np.asarray(probs)
    result = {}
    value_num = len(values)
    sum_combination_perms = list(itertools.product(range(value_num), repeat=k))
    num_poss_keep_dice = scipy.special.binom(n, k)

    for perm in sum_combination_perms:
        cur_sum = 0
        cur_prob = 1
        for i in perm:
            cur_sum += values[i]
            cur_prob *= probs[i]

        prob_rest_dice = get_prob_smaller_than(n-k, values[min(perm)], values, probs)
        cur_prob *= prob_rest_dice * num_poss_keep_dice
        print(cur_sum, prob_rest_dice)

        if cur_sum in result.keys():
            result[cur_sum] += cur_prob
        else:
            result[cur_sum] = cur_prob

    return result


def get_prob_smaller_than(dice_num, value, values, probs):
    values = np.asarray(values)
    probs = np.asarray(probs)

    value_idx = np.where(values == value)[0][0]
    prob_lower = np.sum(probs[:value_idx+1])
    prob_all_lower = prob_lower ** dice_num

    return prob_all_lower


values = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
probs = [1/12]*12

probabilities = n_keep_k_highest(3, 1, values, probs)

mean = 0
for sum, prob in probabilities.items():
    mean += sum * prob
print(np.sum(list(probabilities.values())), mean, probabilities)

plt.plot(probabilities.keys(), probabilities.values())
plt.show()
