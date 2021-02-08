import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def find_gaussian_params(**kwargs):
    target_expected_value = kwargs.pop('expected_value', 6.5)
    target_variance = kwargs.pop('variance', 3)
    values = kwargs.pop('values', np.array(list(range(12)))+1)
    values = np.asarray(values)
    precision = kwargs.pop('precision', 0.01)

    guess_x = target_expected_value
    guess_v = target_variance
    _, initial_guess = generate_gaussian_distribution(center=guess_x, width=guess_v, values=values)
    expected_value, variance, _ = analyse_distribution(values, initial_guess)

    counter = 0
    changed = True
    while changed:
        changed = False
        while abs(target_expected_value - expected_value) > precision:
            changed = True
            guess_off_x = expected_value - target_expected_value
            guess_x -= guess_off_x

            _, guess_distribution = generate_gaussian_distribution(center=guess_x, width=guess_v, values=values)
            expected_value, variance, _ = analyse_distribution(values, guess_distribution)

        while abs(target_variance - variance) > precision:
            changed = True
            guess_off_v = variance - target_variance
            guess_v -= guess_off_v/2

            _, guess_distribution = generate_gaussian_distribution(center=guess_x, width=guess_v, values=values)
            expected_value, variance, _ = analyse_distribution(values, guess_distribution)

        counter += 1
        if counter >= 100:
            print('took too many iterations. Cut-off!')
            break

    return guess_x, guess_v


def generate_gaussian_distribution(center=6.5, width=3, values=None):
    if values is None:
        x = np.linspace(1, 12, 12)
    else:
        x = np.asarray(values)
    y = norm.pdf(x, center, width)
    y /= np.sum(y)

    return x, y


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
    vals = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    center, width = find_gaussian_params(values=vals, expected_value=2.0, variance=4)
    _, probs = generate_gaussian_distribution(center, width, vals)
    mean, std, _ = analyse_distribution(vals, probs)

    print(probs)
    print(mean, std)
    plt.plot(vals, probs)
    plt.show()
