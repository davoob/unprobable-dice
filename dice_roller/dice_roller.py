import pybullet as bt
import time
import pybullet_data
import numpy as np
from structures.polygon import create_dodecahedron, create_cube, Polygon
import pyquaternion as quat
import random
import sys


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


class DiceRoller:
    def __init__(self, die:Polygon):
        self.gravity = [0, 0, -9.81]
        self.cut_off_precision = 0.001

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

    def initialise_simulation(self, run_visible=True, randomize_start=True):
        if run_visible:
            connect_type = bt.GUI
        else:
            connect_type = bt.DIRECT
        self.physics_client_id = bt.connect(connect_type)
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
        self.transfer_mesh()

        # convex mesh from obj
        die_collision_shape = bt.createCollisionShape(bt.GEOM_MESH, fileName="polygon_exchange.obj")
        self.die_id = bt.createMultiBody(baseMass=1,
                           baseCollisionShapeIndex=die_collision_shape,
                           basePosition=self.start_pos,
                           baseOrientation=self.start_rot)
        bt.resetBaseVelocity(self.die_id, linearVelocity=self.start_vel, angularVelocity=self.start_ang)
        bt.changeDynamics(self.die_id, -1, lateralFriction=0.9, spinningFriction=1, restitution=0.1)

    def run(self, visible=True, randomize_start=True):
        self.initialise_simulation(visible, randomize_start)
        for i in range(10000):
            bt.stepSimulation()
            if self.is_settled():
                break
            if visible:
                time.sleep(1. / 240.)
        self.end_pos, self.end_rot = bt.getBasePositionAndOrientation(self.die_id, self.physics_client_id)
        self.end_vel, self.end_ang = bt.getBaseVelocity(self.die_id, self.physics_client_id)
        bt.disconnect(self.physics_client_id)

    def run_multible(self, num=100, debug=False):
        rolled_results = {}
        if debug:
            progress_bar(0, num)
        for run_num in range(num):
            self.run(visible=False)

            up_face_idx = self.find_up_face_idx()
            if up_face_idx in rolled_results.keys():
                rolled_results[up_face_idx] += 1
            else:
                rolled_results[up_face_idx] = 1

            if debug and (run_num+1 % int(num/100) == 0):
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
            self.start_pos = 10 * ([2, 2, 1] * np.random.random(3) + [-1, -1, 0.5])
        if random_vel:
            radius = random.random() * (vel_range[1] - vel_range[0]) + vel_range[0]
            self.start_vel = get_random_vector(r=radius)
        if random_rot:
            self.start_rot = list(get_random_rotation())
        if random_ang:
            radius = random.random() * (ang_range[1] - ang_range[0]) + ang_range[0]
            self.start_ang = get_random_vector(r=radius)

    def transfer_mesh(self):
        self.die.align_normal_to_vector(0, [0, 0, 1])
        self.die.save_obj(file_name='polygon_exchange')

    def find_up_face_idx(self):
        normals = self.die.face_normals

        up_vector = np.asarray(normals[0])
        up_vector_length = np.linalg.norm(up_vector)
        end_rotation = quat.Quaternion(*self.end_rot)

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
        # print(dots)
        return np.argmax(np.asarray(dots))

    def get_up_value(self):
        up_face_idx = self.find_up_face_idx()
        return self.die.face_values[up_face_idx]


if __name__ == '__main__':
    dodeca = create_dodecahedron()
    dodeca.extend_side(0, 0.7)

    roller = DiceRoller(dodeca)
    roller.run(randomize_start=False)
    value_up = roller.get_up_value()
    print(value_up)
    # results = roller.run_multible(10)
    # print(results)
