import sys

import pychrono.core as chrono
import pychrono.irrlicht as chronoirr
from structures.polygon import create_dodecahedron, create_cube, Polygon
import time
import numpy as np


def get_rotaion_quaternion(angle_x, angle_y, angle_z):
    angle_x *= chrono.CH_C_DEG_TO_RAD
    angle_y *= chrono.CH_C_DEG_TO_RAD
    angle_z *= chrono.CH_C_DEG_TO_RAD
    rotation = chrono.Q_from_Euler123(chrono.ChVectorD(angle_x, angle_y, angle_z))
    return rotation


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
        self.container_width = 30
        self.container_length = 30
        self.with_walls = True
        self.wall_height = 50

        self.dice_num = 1

        self.polygon = None
        if type(die_file) is Polygon:
            self.polygon = die_file
            self.die_file = self.polygon.get_chrono_mesh()
        else:
            self.die_file = die_file

        self.cut_off = 0.0001

        # Contact material for container
        self.ground_mat = chrono.ChMaterialSurfaceNSC()
        self.ground_mat.SetFriction(0.8)

        # Shared contact materials for falling objects
        self.dice_mat = chrono.ChMaterialSurfaceNSC()
        self.dice_mat.SetFriction(0.5)

        # initialise variables
        self.system = None
        self.dice = list()

        # init system
        self.initialise_system()

    def initialise_system(self):
        self.system = chrono.ChSystemNSC()

        # Modify some setting of the physical system for the simulation, if you want
        self.system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        # mysystem.SetSolverMaxIterations(20)

        self.add_container()
        self.dice = self.add_dice()

    def reinitialise_system(self):
        del self.system
        del self.dice

        self.initialise_system()

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

    def add_dice(self):
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
            self.set_start_parameters(die_body)
            dice.append(die_body)
        return dice

    def set_start_parameters(self, die):
        position = 10 * ([2, 1, 2] * np.random.random(3) + [-1, 0.5, -1])
        rot_angles = 2*np.pi * np.random.random(3)
        speed = 10 * (2*np.random.random(3)-1)
        ang_speed = 10 * (2*np.random.random(3)-1)

        die.SetPos(chrono.ChVectorD(*position))
        rotation = get_rotaion_quaternion(*rot_angles)
        die.SetRot(rotation)

        die.SetPos_dt(chrono.ChVectorD(*speed))
        die.SetWvel_loc(chrono.ChVectorD(*ang_speed))

        die.SetPos_dtdt(chrono.ChVectorD(0, 0, 0))
        die.SetRot_dtdt(get_rotaion_quaternion(0, 0, 0))

    def run(self):
        # start_t = time.time()
        self.system.SetChTime(0)
        while self.system.GetChTime() < 100:
            self.system.DoStepDynamics(0.02)

            # break if velocity and rotational velocity is below threshold
            if self.dice[0].GetPos_dt().Length() < self.cut_off and \
                    self.dice[0].GetRot_dt().Length() < self.cut_off and \
                    self.dice[0].GetPos().y < 2:
                break
        # end_t = time.time()
        # duration = end_t - start_t
        # rot = chrono.Q_to_Euler123(self.dice[0].GetRot()) * chrono.CH_C_RAD_TO_DEG
        # pos = self.dice[0].GetPos()

        # print(duration, pos, rot)

    def run_multiple(self, num_sim):
        start_t = time.time()
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
        print(counts)
        mean = 0
        for i, count in enumerate(counts):
            mean += count * (i+1)
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
            if self.dice[0].GetPos_dt().Length() < self.cut_off and \
                    self.dice[0].GetRot_dt().Length() < self.cut_off and \
                    self.dice[0].GetPos().y < 2:
                break

        end_t = time.time()
        duration = end_t - start_t
        rot = chrono.Q_to_Euler123(self.dice[0].GetRot()) * chrono.CH_C_RAD_TO_DEG
        pos = self.dice[0].GetPos()

        print(duration, pos, rot)

    def find_up_face_idx(self, normals=None, die_idx=0):
        if normals is None:
            if self.polygon is None:
                print('no normals given')
                return -1
            normals = self.polygon.face_normals

        up_vector = npvec_to_chvec(normals[0])

        die = self.dice[die_idx]
        rotation = die.GetRot()

        ch_normals = npvecs_to_chvecs(normals)
        dots = []
        for i, ch_normal in enumerate(ch_normals):
            # rotate normal
            ch_normal = rotation.Rotate(ch_normal)

            # compare with up_vector
            dot = up_vector ^ ch_normal / (up_vector.Length() * ch_normal.Length())
            dots.append(dot)
            # if round(dot, 1) >= 0.9:
            #     return i
        return np.argmax(np.asarray(dots))


def progress_bar(current, total, bar_length=50):
    percent = float(current) * 100 / total
    arrow = '-' * int(percent/100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write('\r' + 'Progress: [%s%s] %3.2f %%' % (arrow, spaces, percent))


if __name__ == '__main__':
    dodecahedron = create_dodecahedron()
    dodecahedron.align_normal_to_vector(0, [0, 1, 0])

    cube = create_cube()
    cube.align_normal_to_vector(0, [0, 1, 0])

    test_roller = DiceRoller(dodecahedron)
    print(test_roller.system.Get_G_acc())
    # test_roller.run()
    # test_roller.reinitialise_system()
    test_roller.run_visible()
    print(test_roller.find_up_face_idx())
    # test_roller.run_multiple(1000)
