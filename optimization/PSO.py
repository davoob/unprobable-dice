import sys
import numpy as np
from threading import Thread
from multiprocessing import Process, Queue, Pool
from time import sleep
from scipy.stats import norm
import matplotlib.pyplot as plt
from structures.polygon import create_dodecahedron, create_uneven_dodecahedron
import time
from datetime import datetime
import pickle


def load_pso(file_name, **kwargs):
    saved_data = pickle.load(open(file_name + '.pickle', 'rb'))
    init_params = saved_data[0]

    init_params[3] = kwargs.pop('max_gen', init_params[3])
    init_params[5] = kwargs.pop('max_duration', init_params[5])
    init_params[6] = kwargs.pop('num_worker', init_params[6])

    print(len(init_params))
    optimizer = ParticleSwarmOptimization(*init_params)
    cur_gen = saved_data[1]
    optimizer.start_gen = cur_gen + 1
    optimizer.particles_position[:cur_gen+2, :, :] = saved_data[2][:cur_gen+2, :, :]
    optimizer.particles_velocity[:cur_gen+2, :, :] = saved_data[3][:cur_gen+2, :, :]
    optimizer.particles_fitness[:cur_gen+1, :] = saved_data[4][:cur_gen+1, :]
    optimizer.particles_best[:cur_gen+1, :] = saved_data[5][:cur_gen+1, :]
    optimizer.particles_best_position[:cur_gen+1, :, :] = saved_data[6][:cur_gen+1, :, :]
    optimizer.global_best[:cur_gen+1] = saved_data[7][:cur_gen+1]
    optimizer.global_best_position[:cur_gen+1, :] = saved_data[8][:cur_gen+1, :]

    return optimizer


class ParticleSwarmOptimization:
    def __init__(self, fit_func, num_dim, num_particles, max_gen=50, start_gen=0, max_duration=None, num_worker=5,
                 solution_space_limits=None, starting_range_position=None, starting_range_velocity=None,
                 w=0.729, c1=2.0412, c2=0.9477, time_step=1, visualization_func=None, save_name='pso_save'):
        self.fit_func = fit_func
        self.num_dim = num_dim
        self.num_particles = num_particles
        self.max_gen = max_gen
        self.start_gen = start_gen
        self.max_duration = max_duration
        if solution_space_limits is None:
            solution_space_limits = [[0, 1]] * num_dim
        self.solution_space_limits = np.asarray(solution_space_limits)
        self.visualization_func = visualization_func
        self.save_name = save_name

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
        self.starting_range_position = np.asarray(starting_range_position)
        self.starting_range_velocity = np.asarray(starting_range_velocity)

        self.particles_position[0, :, :] = np.random.rand(num_particles, num_dim) * \
                                           (self.starting_range_position[:, 1] - self.starting_range_position[:, 0]) + \
                                           (self.starting_range_position[:, 0])
        self.particles_velocity[0, :, :] = np.random.rand(num_particles, num_dim) * \
                                           (self.starting_range_velocity[:, 1] - self.starting_range_velocity[:, 0]) + \
                                           (self.starting_range_velocity[:, 0])
        self.particles_in_bounds = [True]*num_particles

        self.queue_in = Queue()
        self.queue_out = Queue()

        self.workers = []
        self.num_worker = num_worker
        if num_worker == 1:
            self.multiprocessing = False
        else:
            self.multiprocessing = True
            for worker_id in range(num_worker):
                worker = Process(target=queue_worker, args=(worker_id, self.queue_in, self.queue_out))
                self.workers.append(worker)

        # time keeping
        self.start_time = 0

    def start(self):
        self.start_time = time.time()

        if self.multiprocessing:
            self.connect_workers()

        for gen in range(self.start_gen, self.max_gen):
            if self.multiprocessing:
                self.run_workers(gen)
            else:
                for i in range(self.num_particles):
                    self.particles_fitness[gen, i] = self.fit_func(self.particles_position[gen, i, :].tolist())

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

            # visualize progress
            # visualizing = Thread(target=self.visualize_result, args=(), kwargs={})
            # visualizing.start()
            # visualizing = Pool(processes=1)
            # visualizing.apply_async(self.visualize_result)
            # self.plot_result()

            # test if run exceeds maximum duration
            if self.max_duration is not None:
                if time.time() - self.start_time > self.max_duration * 60:
                    print('\n')
                    self.save_state(gen, self.save_name)
                    break

        print('\n')
        print(self.global_best[gen])

        if self.multiprocessing:
            self.disconnect_workers()

        self.visualize_result()

    def connect_workers(self):
        for worker in self.workers:
            worker.start()

    def run_workers(self, gen):
        # queue jobs for positions of current generation
        worked_particles = 0
        for i in range(self.num_particles):
            if not self.particles_in_bounds[i]:
                worked_particles += 1
            else:
                self.queue_in.put([i, self.fit_func, self.particles_position[gen, i, :].tolist()])

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
            elapsed_time = time.time() - self.start_time
            progress_bar(worked_particles + gen*self.num_particles, self.num_particles*self.max_gen, gen+1, self.global_best[gen-1], elapsed_time)

    def disconnect_workers(self):
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

    def save_state(self, cur_gen, save_name):
        date_time_now = datetime.now()
        time_stamp = date_time_now.strftime("%y%m%d_%H%M%S")

        init_params = [self.fit_func, self.num_dim, self.num_particles, self.max_gen, self.start_gen, self.max_duration,
                       self.num_worker, self.solution_space_limits, self.starting_range_position,
                       self.starting_range_velocity, self.w, self.c1, self.c2, self.time_step, self.visualization_func,
                       self.save_name]
        save_data = [init_params, cur_gen, self.particles_position, self.particles_velocity, self.particles_fitness,
                     self.particles_best, self.particles_best_position, self.global_best, self.global_best_position]

        pickle.dump(save_data, open(save_name + '.pickle', 'wb'))
        print("state saved")

    def visualize_result(self):
        self.plot_result()

        if self.visualization_func is None:
            print('no visualization function')
            return

        best_result = self.get_best_params()
        result_die = create_uneven_dodecahedron(best_result)
        self.visualization_func(result_die)


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


def progress_bar(current, total, generation, global_best, elapsed_time, bar_length=50):
    percent = float(current) * 100 / total
    elapsed_time /= 60  # convert in minutes
    if percent == 0:
        estimated_total_time = -1
    else:
        estimated_total_time = 100/percent * elapsed_time
    remaining_time = (100-percent)/100 * estimated_total_time

    arrow = '-' * int(percent/100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write('\r' + 'Progress: [%s%s] %3.2f %%; generation: %d; global best: %1.4f; elapsed time: %5.2f minutes; remaining time: %5.2f minutes' % (arrow, spaces, percent, generation, global_best, elapsed_time, remaining_time))


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
