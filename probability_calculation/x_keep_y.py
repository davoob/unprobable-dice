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
    crit_prob = 0
    botch_prob = 0
    value_num = len(values)
    # sum_combination_perms = list(itertools.product(range(value_num), repeat=k))
    sum_combination_perms = list(itertools.combinations_with_replacement(range(value_num), k))

    for perm in sum_combination_perms:
        min_value = np.min(values[list(perm)])
        num_dice_equals_min = 0
        cur_sum = 0
        cur_prob = 1
        num_each_value = {}
        is_crit = False
        # is_botch = False
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

            if values[i] == np.max(values):
                is_crit = True
            # if values[i] <= np.min(values):
            #     is_botch = True
        # if is_crit and is_botch:
        #     is_crit = False
        #     is_botch = False

        prob_rest_dice = get_prob_smaller_than(n-k+num_dice_equals_min, values[min(perm)], values, probs,
                                               num_equal_dice=num_dice_equals_min)
        num_poss_keep_dice = 1
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

        if is_crit:
            crit_prob += total_prob
        # if is_botch:
        #     botch_prob += total_prob

    botch_prob = get_prob_smaller_than(int(n/2)+1, 0, values, probs, also_equals=False)
    info = {'crit_prob': crit_prob, 'botch_prob': botch_prob}
    return result, info


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


def analyse_distribution(values, probabilities, test_difficulties=None):
    expected_value = 0
    for value, probability in zip(values, probabilities):
        expected_value += value * probability

    variance = 0
    for value, probability in zip(values, probabilities):
        variance += (value - expected_value)**2 * probability

    other = {}

    if test_difficulties is None:
        test_difficulties = list(range(int(min(values)), int(max(values)) + 3))
    if type(test_difficulties) is not list:
        test_difficulties = [test_difficulties]
    test_difficulties_probabilities = [0]*len(test_difficulties)
    for i, test_difficulty in enumerate(test_difficulties):
        for j, value in enumerate(values):
            if value >= test_difficulty:
                test_difficulties_probabilities[i] += probabilities[j]
    other['test_difficulties_probabilities'] = test_difficulties_probabilities

    half_prob_value = np.interp(0.5, np.flip(np.asarray(test_difficulties_probabilities)), np.flip(np.asarray(test_difficulties)))
    other['half_prob_value'] = half_prob_value

    return expected_value, variance, other


if __name__ == '__main__':
    values = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]) + 4
    probs = [1/12]*12

    uneven_vals_probs = np.loadtxt('distribution.txt')
    uneven_values = uneven_vals_probs[:, 0]
    uneven_probs = uneven_vals_probs[:, 1]
    print(uneven_values, uneven_probs)

    probabilities, info = n_keep_k_highest(3, 1, uneven_values, uneven_probs)

    expect, var, diff_probs = analyse_distribution(list(probabilities.keys()), list(probabilities.values()), test_difficulties=[0, 2, 4, 6, 8, 10, 14, 18, 22, 26, 30])
    print(expect, var, diff_probs)

    plt.plot(probabilities.keys(), probabilities.values())
    plt.show()

