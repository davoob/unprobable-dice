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
    # sum_combination_perms = list(itertools.product(range(value_num), repeat=k))
    sum_combination_perms = list(itertools.combinations_with_replacement(range(value_num), k))

    for perm in sum_combination_perms:
        min_value = np.min(values[list(perm)])
        num_dice_equals_min = 0
        cur_sum = 0
        cur_prob = 1
        num_each_value = {}
        for i in perm:
            cur_sum += values[i]
            if values[i] == min_value:
                num_dice_equals_min += 1
            else:
                cur_prob *= probs[i]
                if values[i] in num_each_value.keys():
                    num_each_value[values[i]] += 1
                else:
                    num_each_value[values[i]] = 1

        prob_rest_dice = get_prob_smaller_than(n-k+num_dice_equals_min, values[min(perm)], values, probs, num_equal_dice=num_dice_equals_min)
        num_poss_keep_dice = 1  # scipy.special.binom(n, k-num_dice_equals_min)
        drawn = 0
        for value in num_each_value.keys():
            num_poss_keep_dice *= scipy.special.binom(n - drawn, num_each_value[value])
            drawn += num_each_value[value]
        total_prob = cur_prob * prob_rest_dice * num_poss_keep_dice
        # print(perm, values[list(perm)], cur_prob, prob_rest_dice, num_poss_keep_dice, num_each_value, num_dice_equals_min)

        if cur_sum in result.keys():
            result[cur_sum] += total_prob
        else:
            result[cur_sum] = total_prob

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


def get_prob_smaller_min_one_equal(dice_num, x, values, probs):
    values = np.asarray(values)
    probs = np.asarray(probs)
    value_idx = np.where(values == x)[0][0]

    prob_all_smaller = get_prob_smaller_than(dice_num, x, values, probs, also_equals=True)

    # transform probs into world with no value higher than x
    values_x = values[:x+1]
    probs_x = probs[:x+1]
    probs_x = probs_x / np.sum(probs_x)

    prob_die_is_smaller = np.sum(probs[:-1])
    prob_min_one_is_equal = 1 - (prob_die_is_smaller ** dice_num)

    return prob_min_one_is_equal


values = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]) + 4
probs = [1/12]*12

uneven_vals_probs = np.loadtxt('distribution.txt')
uneven_values = uneven_vals_probs[:, 0]
uneven_probs = uneven_vals_probs[:, 1]
print(uneven_values, uneven_probs)

probabilities = n_keep_k_highest(3, 1, uneven_values, uneven_probs)

mean = 0
for sum, prob in probabilities.items():
    mean += sum * prob
print(np.sum(list(probabilities.values())), mean, probabilities)

plt.plot(probabilities.keys(), probabilities.values())
plt.show()

