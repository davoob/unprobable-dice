from structures.polygon import create_uneven_dodecahedron
from dice_roller.dice_roller import DiceRoller
from optimization.PSO import ParticleSwarmOptimization
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def generate_gaussian_distribution(center=6.5, width=3):
    x = np.linspace(1, 12, 12)
    y = norm.pdf(x, center, width)
    y /= np.sum(y)

    return x, y


def calculate_max_deviation(distribution, weighting=None, asymmetry_penalty=1):
    distribution_length = len(distribution)
    if weighting is None:
        weighting = np.array([1] * distribution_length)
    # most_off_distribution = np.roll(distribution, int(distribution_length / 2))
    most_off_distribution = np.asarray([0]*(distribution_length-1) + [1])
    deviation = calculate_deviation(most_off_distribution, distribution, weighting, asymmetry_penalty)

    return deviation


def calculate_deviation(distribution, base_distribution, weighting=None, asymmetry_penalty=1):
    distribution_length = len(base_distribution)
    if weighting is None:
        weighting = np.array([1] * distribution_length)
    deviations = np.zeros(distribution_length)
    for i in range(len(deviations)):
        deviations[i] = distribution[i] - base_distribution[i]
    deviation = np.sum(np.square(deviations * weighting))
    asymmetry = 0
    for i in range(int(len(deviations) / 2)):
        asymmetry += asymmetry_penalty * (np.square(deviations[i] - deviations[-(i + 1)]))
    deviation += asymmetry

    return deviation


def get_distribution_from_occurances(occurances_dict: dict, fixed_values=None):
    unsorted_vals = list(occurances_dict.keys())
    unsorted_probs = list(occurances_dict.values())

    if fixed_values is not None:
        for value in fixed_values:
            if value not in unsorted_vals:
                unsorted_vals.append(value)
                unsorted_probs.append(0)
        assert len(fixed_values) == len(unsorted_vals)

    unsorted_vals = np.asarray(unsorted_vals)
    unsorted_probs = np.asarray(unsorted_probs)
    unsorted_probs = unsorted_probs / np.sum(unsorted_probs)

    sorted_idx = unsorted_vals.argsort()
    sorted_vals = unsorted_vals[sorted_idx]
    sorted_probs = unsorted_probs[sorted_idx]

    return sorted_vals, sorted_probs


def run_simulation(die, num_sims=500, debug=False):
    roller = DiceRoller(die, time_step=1./20.)
    result = roller.run_multible(num_sims, debug=debug)
    if debug:
        print(result)
    _, cur_distribution = get_distribution_from_occurances(result, values)
    return cur_distribution


values, gaussian_distribution = generate_gaussian_distribution()
weights = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
max_deviation = calculate_max_deviation(gaussian_distribution, weights)


def simulate_die(params):
    dodecahedron = create_uneven_dodecahedron(params)
    cur_distribution = run_simulation(dodecahedron)
    squared_deviation = calculate_deviation(cur_distribution, gaussian_distribution, weights)
    # print(squared_deviation)

    return (max_deviation - squared_deviation) / max_deviation


if __name__ == '__main__':
    PSO = ParticleSwarmOptimization(simulate_die, 12, 50, solution_space_limits=[[-0.5, 0.5]]*12, max_gen=100,
                                    num_worker=10)
    PSO.start()

    PSO.plot_result()
    best_params = PSO.get_best_params()
    best_die = create_uneven_dodecahedron(best_params)
    best_die.show()

    best_die_distribution = run_simulation(best_die, num_sims=1000, debug=True)

    plt.plot(values, best_die_distribution)
    plt.plot(values, gaussian_distribution)
    plt.show()

    np.savetxt('distribution.txt', np.vstack((values, best_die_distribution, gaussian_distribution)).T,
               header='value probability target_probability')
    best_die.show(show_markings=True, save=True, save_name='bell_dodeka', save_format='obj')
    best_die.save('sim_optimized_dodeca.pickle')