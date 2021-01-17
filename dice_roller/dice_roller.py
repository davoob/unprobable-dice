import pybullet as bt
import time
import pybullet_data
import numpy as np
from structures.polygon import create_dodecahedron, create_cube, Polygon
import pyquaternion as quat
import random
import sys, os
import matplotlib.pyplot as plt


deg_to_rad = np.pi/180
rad_to_deg = 180/np.pi


def get_random_vector(r=0, mode='spherical'):
    if r == 0:
        r = np.cbrt(random.random())
    cos_theta = random.random() * 2 - 1
    theta = np.arccos(cos_theta)
    # theta = random.random() * np.pi
    phi = random.random() * 2 * np.pi
    z = random.random() * 2 - 1

    rand_vector = np.array([0, 0, 0])
    if mode == 'spherical':
        rand_vector = r * np.asarray([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    elif mode == 'cylindrical':
        rand_vector = np.asarray([np.sqrt(1-z**2) * np.cos(phi), np.sqrt(1-z**2) * np.sin(phi), z])
    else:
        print('there is no mode "' + str(mode) + '"')
        raise Exception

    return rand_vector


def get_random_rotation():
    rand_vector = get_random_vector(r=1, mode='spherical')
    rotation = vect_to_rotation(rand_vector)
    return rotation


def vect_to_rotation(vector):
    normal = np.asarray([0, 1, 0])
    if (vector == normal).all():
        rotation = quat.Quaternion(axis=normal, angle=0)
    else:
        rot_vector = np.cross(normal, vector)
        rot_vector = rot_vector / np.linalg.norm(rot_vector)

        angle = np.arccos(np.dot(normal, vector))

        rotation = quat.Quaternion(axis=rot_vector, angle=angle)
    return rotation


def progress_bar(current, total, bar_length=50):
    percent = float(current) * 100 / total
    arrow = '-' * int(percent/100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write('\r' + 'Progress: [%s%s] %3.2f %%' % (arrow, spaces, percent))


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


class DiceRoller:
    def __init__(self, die:Polygon, time_step=1./240.):
        self.gravity = [0, 0, -9.81]
        self.cut_off_precision = 0.01
        self.exchange_id = int(random.random()*1000000)

        self.time_step = time_step

        self.die = die

        self.physics_client_id = 0
        self.plane_id = -1
        self.die_id = -1

        self.start_pos = [0, 0, 5]
        self.start_rot = bt.getQuaternionFromEuler(np.asarray([0, 0, 0])*deg_to_rad)
        self.start_vel = [0, 0, 0]
        self.start_ang = [0, 0, 0]

        self.end_pos = [0, 0, 5]
        self.end_rot = bt.getQuaternionFromEuler(np.asarray([0, 0, 0])*deg_to_rad)
        self.end_vel = [0, 0, 0]
        self.end_ang = [0, 0, 0]

        self.run_history = []
        self.cur_run_num = -1

    def initialise_simulation(self, run_visible=True, randomize_start=True):
        self.cur_run_num += 1
        self.run_history.append({'run_num': self.cur_run_num, 'start_params': [], 'end_params': [], 'up_value': 0})

        if run_visible:
            connect_type = bt.GUI
        else:
            connect_type = bt.DIRECT
        self.physics_client_id = bt.connect(connect_type)
        bt.setPhysicsEngineParameter(fixedTimeStep=self.time_step, numSolverIterations=100)
        # bt.setAdditionalSearchPath(pybullet_data.getDataPath())
        bt.setGravity(self.gravity[0], self.gravity[1], self.gravity[2], self.physics_client_id)

        plane_id = bt.createCollisionShape(bt.GEOM_PLANE, planeNormal=[0, 0, 1])
        self.plane_id = bt.createMultiBody(baseMass=0,
                                           baseCollisionShapeIndex=plane_id)
        bt.changeDynamics(self.plane_id, -1, lateralFriction=0.9, spinningFriction=1, restitution=0.1)
        # self.plane_id = bt.loadURDF("plane.urdf")

        if randomize_start:
            # randomize start parameters
            self.random_start()

        # get obj from die_polygon
        exchange_filename = self.transfer_mesh()

        # convex mesh from obj
        die_collision_shape = bt.createCollisionShape(bt.GEOM_MESH,
                                                      fileName=exchange_filename,
                                                      meshScale=0.01)
        os.remove(exchange_filename)

        self.die_id = bt.createMultiBody(baseMass=1,
                                         baseCollisionShapeIndex=die_collision_shape,
                                         basePosition=self.start_pos,
                                         baseOrientation=self.start_rot)
        bt.resetBaseVelocity(self.die_id, linearVelocity=self.start_vel, angularVelocity=self.start_ang)
        bt.changeDynamics(self.die_id, -1, lateralFriction=0.9, spinningFriction=1, restitution=0.1)

        self.run_history[self.cur_run_num]['start_params'] = [self.start_pos, self.start_vel, self.start_rot, self.start_ang]

    def run(self, visible=True, randomize_start=True):
        self.initialise_simulation(visible, randomize_start)
        for i in range(10000):
            bt.stepSimulation()
            if self.is_settled():
                break
            if visible:
                time.sleep(self.time_step/2)
        self.end_pos, self.end_rot = bt.getBasePositionAndOrientation(self.die_id, self.physics_client_id)
        self.end_vel, self.end_ang = bt.getBaseVelocity(self.die_id, self.physics_client_id)
        self.run_history[self.cur_run_num]['end_params'] = [self.end_pos, self.end_vel, self.end_rot, self.end_ang]
        bt.disconnect(self.physics_client_id)

    def run_multible(self, num=100, debug=False):
        rolled_results = {}
        if debug:
            progress_bar(0, num)
        for run_num in range(num):
            self.run(visible=False)

            up_face_value = self.get_up_value()
            if up_face_value in rolled_results.keys():
                rolled_results[up_face_value] += 1
            else:
                rolled_results[up_face_value] = 1

            if debug and (run_num % 5 == 0):
                progress_bar(run_num+1, num)

        return rolled_results

    def is_settled(self, debug=False):
        pos, rot = bt.getBasePositionAndOrientation(self.die_id, self.physics_client_id)
        vel, ang = bt.getBaseVelocity(self.die_id, self.physics_client_id)

        if debug:
            print(pos, rot, vel, ang)

        is_settled = np.linalg.norm(vel) < self.cut_off_precision and \
                     np.linalg.norm(ang) < self.cut_off_precision and \
                     pos[2] < 5

        return is_settled

    def random_start(self, **kwargs):
        random_pos = kwargs.pop('random_pos', False)
        random_vel = kwargs.pop('random_vel', True)
        random_rot = kwargs.pop('random_rot', True)
        random_ang = kwargs.pop('random_ang', True)

        vel_range = kwargs.pop('vel_range', [0.5, 10])
        ang_range = kwargs.pop('ang_range', [0.5, 10])

        if random_pos:
            self.start_pos = 0.1 * ([2, 2, 1] * np.random.random(3) + [-1, -1, 0.5])
        if random_vel:
            radius = random.random() * (vel_range[1] - vel_range[0]) + vel_range[0]
            self.start_vel = get_random_vector(r=radius)
        if random_rot:
            self.start_rot = list(get_random_rotation())
        if random_ang:
            radius = random.random() * (ang_range[1] - ang_range[0]) + ang_range[0]
            self.start_ang = get_random_vector(r=radius)

    def transfer_mesh(self):
        filename = 'polygon_exchange_' + str(self.exchange_id)
        self.die.align_normal_to_vector(0, [0, 0, 1])
        self.die.save_obj(file_name=filename)
        return filename + '.obj'

    def find_up_face_idx(self, debug=False):
        normals = self.die.face_normals

        up_vector = np.asarray(normals[0])
        up_vector_length = np.linalg.norm(up_vector)
        end_rotation = quat.Quaternion(self.end_rot[3], self.end_rot[0], self.end_rot[1], self.end_rot[2])

        dots = []
        for i, normal in enumerate(normals):
            # rotate normal
            end_normal = end_rotation.rotate(np.asarray(normal))
            end_normal_length = np.linalg.norm(end_normal)

            # compare rotated normal with up_vector
            dot = np.dot(up_vector, end_normal) / (up_vector_length * end_normal_length)
            dots.append(dot)
            # if round(dot, 1) == 0.9:
            #     return i
        if debug:
            print(dots)
        return np.argmax(np.asarray(dots))

    def get_up_value(self, debug=False):
        up_face_idx = self.find_up_face_idx(debug)
        up_value = self.die.face_values[up_face_idx]
        self.run_history[self.cur_run_num]['up_value'] = up_value
        return up_value


if __name__ == '__main__':
    """ up value testing """
    # dodeca = create_dodecahedron()
    # dodeca.extend_side(9, 0.8)
    #
    # roller = DiceRoller(dodeca, time_step=1./20.)
    # roller.start_pos = [0, 0, 2.5]
    # roller.start_rot = bt.getQuaternionFromEuler(np.asarray([180, 0, 0]) * deg_to_rad)
    # roller.start_vel = [0, 0, 0]
    # roller.start_ang = [0, 0, 0]
    # roller.run(visible=True, randomize_start=False)
    # value_up = roller.get_up_value(debug=True)
    # print(value_up)

    """ rerun """
    dodeca = create_dodecahedron()
    dodeca.extend_side(9, 0.25)

    roller = DiceRoller(dodeca, time_step=1./20.)
    roller.run(visible=True)
    value_up = roller.get_up_value()
    print(value_up)

    start_time = time.time()
    results = roller.run_multible(500, debug=True)
    duration = time.time() - start_time
    values, probabilities = get_distribution_from_occurances(results, dodeca.face_values)
    print(duration, results)

    plt.plot(values, probabilities)
    plt.show()

    min_prob_idx = np.argmin(probabilities)
    min_prob_value = values[min_prob_idx]
    print(min_prob_value)
    for run in roller.run_history:
        if run['up_value'] == 1:
            [pos, vel, rot, ang] = run['start_params']
            roller.start_pos = pos
            roller.start_vel = vel
            roller.start_rot = rot
            roller.start_ang = ang

            roller.run(visible=True, randomize_start=False)
            print(roller.get_up_value(debug=True))

            break

    """ Deformation series test """
    # extension_factors = np.array(list(range(15)))
    # extension_step = 0.1
    # min_extension = -0.7
    # extensions = min_extension + extension_factors * extension_step
    #
    # means = []
    # vars = []
    # durations = []
    #
    # for extension in extensions:
    #     print('start testing with extension ' + str(extension))
    #     dodeca = create_dodecahedron()
    #
    #     dodeca.extend_side(0, extension)
    #
    #     roller = DiceRoller(dodeca, time_step=1./20.)
    #     # roller.run(randomize_start=False)
    #     # value_up = roller.get_up_value()
    #     # print(value_up)
    #     start_time = time.time()
    #     results = roller.run_multible(5000, debug=True)
    #     duration = time.time() - start_time
    #     values, probabilities = get_distribution_from_occurances(results, dodeca.face_values)
    #     durations.append(duration)
    #
    #     mean = 0
    #     for value, probability in zip(values, probabilities):
    #         mean += value * probability
    #     means.append(mean)
    #
    #     var = 0
    #     for value, probability in zip(values, probabilities):
    #         var += (value - mean)**2 * probability
    #     var = np.sqrt(var)
    #     vars.append(var)
    #
    #     print(' ; done in ' + str(round(duration)) + 's')
    #     plt.plot(values, probabilities, label='time step: ' + str(extension))
    # plt.legend()
    # plt.show()
    #
    # print('mean duration: ' + str(np.sum(durations)/len(durations)))
    #
    # plt.plot(extensions, means)
    # plt.plot(extensions, vars)
    # plt.show()

    """ Time step test """
    # dodeca = create_dodecahedron()
    # dodeca.extend_side(0, 0.5)
    # roller = DiceRoller(dodeca)
    #
    # time_step_factors = np.array(list(range(15)))+1
    # min_time_step = 1./240.
    #
    # durations = []
    # means = []
    # vars = []
    # for time_step_factor in time_step_factors:
    #     time_step = min_time_step * time_step_factor
    #     roller.time_step = time_step
    #
    #     start_time = time.time()
    #     results = roller.run_multible(500, debug=True)
    #     duration = time.time() - start_time
    #     values, probabilities = get_distribution_from_occurances(results, dodeca.face_values)
    #
    #     mean = 0
    #     for value, probability in zip(values, probabilities):
    #         mean += value * probability
    #
    #     var = 0
    #     for value, probability in zip(values, probabilities):
    #         var += (value - mean)**2 * probability
    #     var = np.sqrt(var)
    #
    #     durations.append(duration)
    #     means.append(mean)
    #     vars.append(var)
    #
    #     # print('\n')
    #     # print(duration, mean, var)
    #
    #     # plt.plot(values, probabilities, label='time step: ' + str(time_step))
    #
    # plt.plot(time_step_factors*min_time_step, durations, label='durations')
    # plt.show()
    #
    # plt.plot(time_step_factors*min_time_step, vars, label='vars')
    # plt.plot(time_step_factors*min_time_step, means, label='means')
    # plt.legend()
    # plt.show()
