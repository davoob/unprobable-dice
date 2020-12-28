import matplotlib.pyplot as plt
from probability_calculation.x_keep_y import n_keep_k_highest, analyse_distribution
import numpy as np

uneven_vals_probs = np.loadtxt('distribution.txt')
values = uneven_vals_probs[:, 0] - 4
probs = uneven_vals_probs[:, 1]
# plt.plot(values, probs)
# plt.show()

nums_extra_dice = list(range(10))
nums_base_dice = list(range(1, 5))
test_difficulties = list(range(int(min(values)), int(max(values)*max(nums_base_dice))+3))

half_prob_values = []
for num_base_dice in nums_base_dice:
    cur_half_prob_values = []
    for num_extra_dice in nums_extra_dice:
        result = n_keep_k_highest(num_base_dice + num_extra_dice, num_base_dice, values, probs)
        expected, var, other = analyse_distribution(list(result.keys()), list(result.values()), test_difficulties)

        # print(expected, var, other['half_prob_value'])
        cur_half_prob_values.append(other['half_prob_value'])
        plt.plot(test_difficulties, other['test_difficulties_probabilities'], label=str(num_extra_dice))
    plt.legend()
    plt.show()
    half_prob_values.append(cur_half_prob_values)

probs = [1/12]*12

half_prob_values_even = []
for num_base_dice in nums_base_dice:
    cur_half_prob_values = []
    for num_extra_dice in nums_extra_dice:
        result = n_keep_k_highest(num_base_dice + num_extra_dice, num_base_dice, values, probs)
        expected, var, other = analyse_distribution(list(result.keys()), list(result.values()), test_difficulties)

        # print(expected, var, other['half_prob_value'])
        cur_half_prob_values.append(other['half_prob_value'])
        plt.plot(test_difficulties, other['test_difficulties_probabilities'], label=str(num_extra_dice))
    plt.legend()
    plt.show()
    half_prob_values_even.append(cur_half_prob_values)

for i, cur_half_prob_values in enumerate(half_prob_values):
    plt.plot(nums_extra_dice, cur_half_prob_values, label=str(nums_base_dice[i]))
for i, cur_half_prob_values in enumerate(half_prob_values_even):
    plt.plot(nums_extra_dice, cur_half_prob_values, '--', label=str(nums_base_dice[i]))
plt.legend()
plt.show()
