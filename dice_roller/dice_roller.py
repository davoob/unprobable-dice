import sys

import pychrono.core as chrono
import pychrono.irrlicht as chronoirr
from structures.polygon import create_dodecahedron, create_cube, Polygon
import time
import numpy as np
import random


def get_rotation_quaternion(angle_x, angle_y, angle_z):
    angle_x *= chrono.CH_C_DEG_TO_RAD
    angle_y *= chrono.CH_C_DEG_TO_RAD
    angle_z *= chrono.CH_C_DEG_TO_RAD
    rotation = chrono.Q_from_Euler123(chrono.ChVectorD(angle_x, angle_y, angle_z))
    return rotation


def vect_to_rotation(vector):
    normal = np.asarray([0, 1, 0])
    if (vector == normal).all():
        rotation = chrono.Q_from_AngAxis(0, chrono.ChVectorD(*normal))
    else:
        rot_vector = np.cross(normal, vector)
        rot_vector = rot_vector / np.linalg.norm(rot_vector)

        angle = np.arccos(np.dot(normal, vector))

        rotation = chrono.Q_from_AngAxis(angle, chrono.ChVectorD(*rot_vector))
    return rotation


def get_random_rotation():
    rand_vector = get_random_vector(r=1)
    rotation = vect_to_rotation(rand_vector)
    return rotation


def get_random_vector(r=0):
    if r == 0:
        r = np.cbrt(random.random())
    cos_theta = random.random() * 2 - 1
    theta = np.arccos(cos_theta)
    # theta = random.random() * np.pi
    phi = random.random() * 2 * np.pi
    z = random.random() * 2 - 1
    rand_vector_sphaeric = r * np.asarray([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    rand_vector_cylindric = np.asarray([np.sqrt(1-z**2) * np.cos(phi), np.sqrt(1-z**2) * np.sin(phi), z])

    return rand_vector_sphaeric


def npvec_to_chvec(npvec):
    chvec = chrono.ChVectorD(*npvec.tolist())
    return chvec


def npvecs_to_chvecs(npvecs):
    chvecs = []
    for npvec in npvecs:
        chvecs.append(npvec_to_chvec(npvec))
    return chvecs


class DiceRoller:
    def __init__(self, die_file='die.obj'):
        self.container_width = 200
        self.container_length = 200
        self.with_walls = True
        self.wall_height = 50

        self.dice_num = 1

        self.polygon = None
        self.normal_rotation_offset = chrono.Q_from_AngAxis(0, chrono.VECT_Z)
        if type(die_file) is Polygon:
            self.polygon = die_file
            self.die_file = self.polygon.get_chrono_mesh()
            angle, axis = self.polygon.align_normal_to_vector(0, [0, 1, 0], only_info=True)
            self.normal_rotation_offset = chrono.Q_from_AngAxis(angle, chrono.ChVectorD(*axis))
        else:
            self.die_file = die_file

        self.cut_off = 0.0001

        # Contact material for container
        self.ground_mat = chrono.ChMaterialSurfaceNSC()
        self.ground_mat.SetFriction(0.8)

        # Shared contact materials for falling objects
        self.dice_mat = chrono.ChMaterialSurfaceNSC()
        self.dice_mat.SetFriction(0.5)

        # initialise start parameters
        self.dice_position = [0, 5, 0]
        self.dice_speed = [0, 0, 0]
        self.dice_rotation = [0, 0, 0]
        self.dice_ang_speed = [0, 0, 0]
        self.past_start_params = []
        self.run_results = []

        # initialise other variables
        self.system = None
        self.dice = list()

        # init system
        self.initialise_system()

    def initialise_system(self, use_set_params=False):
        self.system = chrono.ChSystemNSC()

        # Modify some setting of the physical system for the simulation, if you want
        self.system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        # mysystem.SetSolverMaxIterations(20)

        self.add_container()
        self.dice = self.add_dice(use_set_params)

        # adding start up facing side to past_start_params
        start_most_up_side_idx = self.find_up_face_idx()
        self.past_start_params[-1] += (start_most_up_side_idx,)

    def reinitialise_system(self, use_set_params=False):
        del self.system
        del self.dice

        self.initialise_system(use_set_params)

    def add_container(self):
        padding = 10

        # Create the five walls of the rectangular container, using fixed rigid bodies of 'box' type
        floor_body = chrono.ChBodyEasyBox(self.container_width+padding, 1, self.container_length+padding,
                                          1000, True, True, self.ground_mat)
        floor_body.SetPos(chrono.ChVectorD(0, 0, 0))
        floor_body.SetBodyFixed(True)
        self.system.Add(floor_body)

        if self.with_walls:
            wall_body_1 = chrono.ChBodyEasyBox(1, self.wall_height, self.container_length+padding,
                                               1000, True, True, self.ground_mat)
            wall_body_1.SetPos(chrono.ChVectorD(-self.container_width/2, self.wall_height/2, 0))
            wall_body_1.SetBodyFixed(True)
            self.system.Add(wall_body_1)

            wall_body_2 = chrono.ChBodyEasyBox(1, self.wall_height, self.container_length+padding,
                                               1000, True, True, self.ground_mat)
            wall_body_2.SetPos(chrono.ChVectorD(self.container_width/2, self.wall_height/2, 0))
            wall_body_2.SetBodyFixed(True)
            self.system.Add(wall_body_2)

            wall_body_3 = chrono.ChBodyEasyBox(self.container_width+padding, self.wall_height, 1,
                                               1000, False, True, self.ground_mat)
            wall_body_3.SetPos(chrono.ChVectorD(0, self.wall_height/2, -self.container_length/2))
            wall_body_3.SetBodyFixed(True)
            self.system.Add(wall_body_3)

            wall_body_4 = chrono.ChBodyEasyBox(self.container_width+padding, self.wall_height, 1,
                                               1000, True, True, self.ground_mat)
            wall_body_4.SetPos(chrono.ChVectorD(0, self.wall_height/2, self.container_length/2))
            wall_body_4.SetBodyFixed(True)
            self.system.Add(wall_body_4)

    def add_dice(self, use_set_params=False):
        dice = list()

        # Create falling rigid bodies (spheres and boxes etc.)
        for i in range(self.dice_num):
            die_body = chrono.ChBodyEasyMesh(self.die_file,    # obj filename
                                             10000,               # density
                                             True,              # compute mass?
                                             True,              # visualization?
                                             True,              # collision?
                                             self.dice_mat)     # material
            self.system.Add(die_body)
            self.set_start_parameters(die_body, use_set_params)
            dice.append(die_body)
        return dice

    def set_start_parameters(self, die, use_set_params=False):
        if not use_set_params:
            self.dice_position = 10 * ([2, 1, 2] * np.random.random(3) + [-1, 0.5, -1])
            self.dice_rotation = get_random_rotation()  # 360 * np.random.random(3)
            self.dice_speed = 10 * get_random_vector()
            self.dice_ang_speed = 10 * get_random_vector()

        die.SetPos(chrono.ChVectorD(*self.dice_position))
        # rotation = get_rotation_quaternion(*self.dice_rotation)
        die.SetRot(self.dice_rotation)

        die.SetPos_dt(chrono.ChVectorD(*self.dice_speed))
        # die.SetRot_dt(self.dice_ang_speed)
        # die.SetWvel_loc(chrono.ChVectorD(*self.dice_ang_speed))

        die.SetPos_dtdt(chrono.ChVectorD(0, 0, 0))
        die.SetRot_dtdt(get_rotation_quaternion(0, 0, 0))

        self.past_start_params.append((self.dice_position, self.dice_speed, self.dice_rotation, self.dice_ang_speed))

    def reset_run_history(self):
        self.past_start_params = []

    def is_settled(self):
        is_settled = self.dice[0].GetPos_dt().Length() < self.cut_off and \
                     self.dice[0].GetRot_dt().Length() < self.cut_off and \
                     self.dice[0].GetPos().y < 2
        return is_settled

    def run(self):
        start_t = time.time()
        self.system.SetChTime(0)
        while self.system.GetChTime() < 100:
            self.system.DoStepDynamics(0.02)

            # break if velocity and rotational velocity is below threshold
            if self.is_settled():
                break
        end_t = time.time()
        duration = end_t - start_t

        self.post_run(duration)

    def run_multiple(self, num_sim):
        start_t = time.time()
        if self.polygon:
            counts = [0]*len(self.polygon.face_values)
        else:
            counts = [0]*12
        progress_bar(0, num_sim-1)
        for i in range(num_sim):
            self.run()
            face_idx = self.find_up_face_idx()
            if face_idx == -1:
                print('no result')
            else:
                counts[face_idx] += 1
            self.reinitialise_system()
            progress_bar(i, num_sim-1)
        print('\n')
        end_t = time.time()
        duration = end_t - start_t
        print(duration)
        counts /= np.sum(counts)
        print(counts, self.polygon.face_values)
        mean = 0
        for i, count in enumerate(counts):
            mean += count * self.polygon.face_values[i]
        print(mean)

    def run_visible(self):
        visible_sim = chronoirr.ChIrrApp(self.system, 'Falling', chronoirr.dimension2du(1024, 768))

        # visible_sim.AddTypicalSky()
        # visible_sim.AddTypicalLogo(chrono.GetChronoDataFile('logo_pychrono_alpha.png'))
        visible_sim.AddTypicalCamera(chronoirr.vector3df(0, 14, -20))
        visible_sim.AddTypicalLights()

        visible_sim.AssetBindAll()
        visible_sim.AssetUpdateAll()
        visible_sim.SetTimestep(0.02)
        visible_sim.SetTryRealtime(True)

        start_t = time.time()
        while visible_sim.GetDevice().run():
            visible_sim.BeginScene()
            visible_sim.DrawAll()
            visible_sim.DoStep()
            visible_sim.EndScene()

            # break if velocity and rotational velocity is below threshold
            if self.is_settled():
                break

        end_t = time.time()
        duration = end_t - start_t

        self.post_run(duration, silent=False)

    def post_run(self, duration, silent=True):
        pos = self.dice[0].GetPos()
        vel = self.dice[0].GetPos_dt()
        rot = chrono.Q_to_Euler123(self.dice[0].GetRot() * self.normal_rotation_offset) * chrono.CH_C_RAD_TO_DEG
        ang_vel = self.dice[0].GetWvel_loc()

        up_side = self.find_up_face_idx()
        up_value = -1
        if up_side != -1:
            up_value = self.polygon.face_values[up_side]
        result = (pos, vel, rot, ang_vel, up_value)

        self.run_results.append(result)
        if not silent:
            print(duration, pos, rot)

    def show_run(self, run_idx):
        run_params = self.past_start_params[run_idx]

        # setup parameters from past run
        self.dice_position = run_params[0]
        self.dice_speed = run_params[1]
        self.dice_rotation = run_params[2]
        self.dice_ang_speed = run_params[3]

        self.reinitialise_system(use_set_params=True)
        self.run_visible()

    def find_in_past_runs(self, value):
        runs_idx = [i for i, item in enumerate(self.run_results) if item[4] == value]
        return runs_idx

    def find_up_face_idx(self, normals=None, die_idx=0):
        if normals is None:
            if self.polygon is None:
                print('no normals given')
                return -1
            normals = self.polygon.face_normals

        up_vector = npvec_to_chvec(normals[0])
        up_vector = self.normal_rotation_offset.Rotate(up_vector)

        die = self.dice[die_idx]
        rotation = die.GetRot()

        ch_normals = npvecs_to_chvecs(normals)
        dots = []
        for i, ch_normal in enumerate(ch_normals):
            # rotate normal
            ch_normal = self.normal_rotation_offset.Rotate(ch_normal)
            ch_normal = rotation.Rotate(ch_normal)

            # compare with up_vector
            dot = up_vector ^ ch_normal / (up_vector.Length() * ch_normal.Length())
            dots.append(dot)
            if round(dot, 1) == 1:
                return i
        # print(dots)
        return np.argmax(np.asarray(dots))


def progress_bar(current, total, bar_length=50):
    percent = float(current) * 100 / total
    arrow = '-' * int(percent/100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write('\r' + 'Progress: [%s%s] %3.2f %%' % (arrow, spaces, percent))


if __name__ == '__main__':
    dodecahedron = create_dodecahedron()
    # dodecahedron.align_normal_to_vector(0, [0, 1, 0])

    cube = create_cube()
    # cube.scale_face(0, 1.0)
    # cube.align_normal_to_vector(3, [0, 1, 0])

    test_roller = DiceRoller(dodecahedron)
    # print(test_roller.find_up_face_idx())
    # test_roller.run()
    # test_roller.reinitialise_system()
    # test_roller.run_visible()
    # print(test_roller.find_up_face_idx()+1)
    test_roller.run_multiple(1000)

    start_values = [test_roller.polygon.face_values[item[-1]] for item in test_roller.past_start_params]
    value_probs = [start_values.count(item) for item in range(1, len(test_roller.polygon.face_values) + 1)]
    value_probs = np.asarray(value_probs) / len(start_values)
    print(value_probs.round(3))

    # runs_with_1 = test_roller.find_in_past_runs(value=1)
    # print(len(runs_with_1), runs_with_1)
    # runs_with_2 = test_roller.find_in_past_runs(value=2)
    # print(len(runs_with_2), runs_with_2)
    # runs_with_12 = test_roller.find_in_past_runs(value=12)
    # print(len(runs_with_12), runs_with_12)
    # test_roller.show_run(runs_with_1[0])
    # test_roller.show_run(runs_with_1[1])
    # test_roller.show_run(runs_with_1[2])
