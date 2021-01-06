import sys
import numpy as np
# from threading import Event, Thread, Lock
from multiprocessing import Process, Queue
# from queue import Queue
from time import sleep
from scipy.stats import norm
import matplotlib.pyplot as plt
from structures.polygon import create_dodecahedron


class ParticleSwarmOptimization:
    def __init__(self, fit_func, num_dim, num_particles, max_gen=50, solution_space_limits=None,
                 starting_range_position=None, starting_range_velocity=None, num_worker=5,
                 w=0.729, c1=2.0412, c2=0.9477, time_step=1):
        self.fit_func = fit_func
        self.num_dim = num_dim
        self.num_particles = num_particles
        self.max_gen = max_gen
        if solution_space_limits is None:
            solution_space_limits = [[0, 1]] * num_dim
        self.solution_space_limits = np.asarray(solution_space_limits)

        # weights
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.time_step = time_step

        # initialize arrays
        self.particles_position = np.zeros((max_gen, num_particles, num_dim))
        self.particles_velocity = np.zeros((max_gen, num_particles, num_dim))
        self.particles_fitness = np.zeros((max_gen, num_particles))
        self.particles_best = np.zeros((max_gen, num_particles))
        self.particles_best_position = np.zeros((max_gen, num_particles, num_dim))
        self.global_best = np.zeros(max_gen)
        self.global_best_position = np.zeros((max_gen, num_dim))

        # set starting positions and velocities
        if starting_range_position is None:
            starting_range_position = solution_space_limits
        if starting_range_velocity is None:
            starting_range_velocity = solution_space_limits
        starting_range_position = np.asarray(starting_range_position)
        starting_range_velocity = np.asarray(starting_range_velocity)

        self.particles_position[0, :, :] = np.random.rand(num_particles, num_dim) * \
                                           (starting_range_position[:, 1] - starting_range_position[:, 0]) + \
                                           (starting_range_position[:, 0])
        self.particles_velocity[0, :, :] = np.random.rand(num_particles, num_dim) * \
                                           (starting_range_velocity[:, 1] - starting_range_velocity[:, 0]) + \
                                           (starting_range_velocity[:, 0])
        self.particles_in_bounds = [True]*num_particles

        self.queue_in = Queue()
        self.queue_out = Queue()

        self.workers = []
        for worker_id in range(num_worker):
            worker = Process(target=queue_worker, args=(worker_id, self.queue_in, self.queue_out))
            self.workers.append(worker)

    def start(self):
        for worker in self.workers:
            worker.start()

        for gen in range(self.max_gen):
            # queue jobs for positions of current generation
            worked_particles = 0
            for i in range(self.num_particles):
                if not self.particles_in_bounds[i]:
                    worked_particles += 1
                else:
                    self.queue_in.put([i, gauss_dodecahedron, self.particles_position[gen, i, :].tolist()])

            # work off all jobs
            break_counter = 0
            while not worked_particles == self.num_particles:
                if not self.queue_out.empty():
                    result = self.queue_out.get()
                    idx = result[0]
                    fitness = result[1]
                    self.particles_fitness[gen, idx] = fitness
                    worked_particles += 1
                else:
                    sleep(0.5)
                    break_counter += 1
                    if break_counter > 1000:
                        break
                progress_bar(worked_particles + gen*self.num_particles, self.num_particles*self.max_gen, gen+1, self.global_best[gen-1])

            # record personal and global bests
            for particle_idx in range(self.num_particles):
                fitness_history = self.particles_fitness[:, particle_idx]
                self.particles_best[gen, particle_idx] = np.max(fitness_history)
                self.particles_best_position[gen, particle_idx, :] = self.particles_position[np.argmax(fitness_history), particle_idx, :]
            self.global_best[gen] = np.max(self.particles_best[gen, :])
            self.global_best_position[gen] = self.particles_best_position[gen, np.argmax(self.particles_best[gen, :]), :]

            # stop if it's the last generation
            if gen+1 == self.max_gen:
                break

            # update every particle's velocity and position for next generation
            for particle_idx in range(self.num_particles):
                cur_pos = self.particles_position[gen, particle_idx, :]
                cur_vel = self.particles_velocity[gen, particle_idx, :]
                new_vel = self.w * cur_vel + self.c1 * np.random.rand(self.num_dim) *\
                                             (self.particles_best_position[gen, particle_idx, :] - cur_pos) +\
                          self.c2 * np.random.rand(self.num_dim) * (self.global_best_position[gen, :] - cur_pos)

                new_pos = cur_pos + self.time_step * new_vel

                self.particles_velocity[gen+1, particle_idx, :] = new_vel
                self.particles_position[gen+1, particle_idx, :] = new_pos

                # check if particle is in solution space
                self.particles_in_bounds[particle_idx] = True
                for dim in range(self.num_dim):
                    in_bounds = self.solution_space_limits[dim][0] <= new_pos[dim] <= self.solution_space_limits[dim][1]
                    if not in_bounds:
                        self.particles_in_bounds[particle_idx] = False

        print('\n')
        print(self.global_best[-1])

        for _ in self.workers:
            self.queue_in.put('STOP')
        for worker in self.workers:
            worker.join()

    def plot_result(self):
        gens = np.linspace(0, self.max_gen - 1, self.max_gen)
        for i in range(self.num_particles):
            particle_best = self.particles_best[:, i]
            plt.plot(gens, particle_best)
        plt.plot(gens, self.global_best)
        plt.show()

    def get_best_params(self):
        return self.global_best_position[-1, :]


def queue_worker(worker_id, queue_in, queue_out):
    # Read from the queue; this will be spawned as a separate Process
    while True:
        msg = queue_in.get()         # Read from the queue and process
        if msg == 'STOP':
            break
        else:
            idx = msg[0]
            func = msg[1]
            params = msg[2]
            result = func(params)
            queue_out.put([idx, result, worker_id])


def gauss_dodecahedron(params):
    dodecahedron = create_dodecahedron()
    for idx, param in enumerate(params):
        dodecahedron.extend_side(idx, param)

    dodecahedron.calculate_anisotropic_trapping_angle(angle_points=720)
    dodecahedron.limit_trapping_ranges_opposite()
    dodecahedron.calculate_trapping_areas()
    dodecahedron.calculate_probabilities()

    values = np.linspace(1, 12, 12)
    gaussian_distribution = norm.pdf(values, 6.5, 3)
    gaussian_distribution /= np.sum(gaussian_distribution)
    weights = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    most_off_distribution = np.roll(gaussian_distribution, 6)
    deviations = np.zeros(len(gaussian_distribution))
    for i in range(len(deviations)):
        deviations[i] = most_off_distribution[i] - gaussian_distribution[i]
    max_deviation = np.sum(np.square(deviations*weights))
    asymmetry = 0
    for i in range(int(len(deviations)/2)):
        asymmetry += np.square(deviations[i]-deviations[-(i+1)])
    max_deviation += asymmetry

    deviations = dodecahedron.compare_probability_to(gaussian_distribution)
    squared_deviation = np.sum(np.square(deviations*weights))
    asymmetry = 0
    for i in range(int(len(deviations)/2)):
        asymmetry += np.square(deviations[i]-deviations[-(i+1)])
    squared_deviation += asymmetry

    return (max_deviation - squared_deviation) / max_deviation


def progress_bar(current, total, generation, global_best, bar_length=50):
    percent = float(current) * 100 / total
    arrow = '-' * int(percent/100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write('\r' + 'Progress: [%s%s] %3.2f %%; generation: %d; global best: %1.3f' % (arrow, spaces, percent, generation, global_best))


if __name__ == '__main__':
    PSO = ParticleSwarmOptimization(gauss_dodecahedron, 12, 100, solution_space_limits=[[-0.5, 0.5]]*12, max_gen=500,
                                    num_worker=10)
    PSO.start()

    generations = np.linspace(0, PSO.max_gen-1, PSO.max_gen)
    for ii in range(PSO.num_particles):
        personal_best = PSO.particles_best[:, ii]
        plt.plot(generations, personal_best)
    plt.plot(generations, PSO.global_best)
    plt.show()

    params = PSO.global_best_position[-1, :]
    print(params)
    final_dodeca = create_dodecahedron()
    for idx, param in enumerate(params):
        final_dodeca.extend_side(idx, param)

    final_dodeca.calculate_anisotropic_trapping_angle(angle_points=720)
    final_dodeca.limit_trapping_ranges_opposite()
    final_dodeca.calculate_trapping_areas()
    final_dodeca.calculate_probabilities()

    values, probs = final_dodeca.get_probabilities()
    gaussian_distribution = norm.pdf(values, 6.5, 3)
    gaussian_distribution /= np.sum(gaussian_distribution)
    squared_deviation = final_dodeca.compare_probability_to(gaussian_distribution)
    print(squared_deviation)

    plt.plot(values, probs)
    plt.plot(values, gaussian_distribution)
    plt.show()

    np.savetxt('distribution.txt', np.vstack((values, probs, gaussian_distribution)).T,
               header='value probability target_probability')
    final_dodeca.show(show_markings=True, save=True, save_name='bell_dodeka', save_format='obj')
    final_dodeca.save('optimized_dodeca.pickle')
