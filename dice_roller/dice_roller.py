import pybullet as bt
import time
import pybullet_data
import numpy as np
from structures.polygon import create_dodecahedron, create_cube, Polygon
import pyquaternion as quat


deg_to_rad = np.pi/180
rad_to_deg = 180/np.pi


class DiceRoller:
    def __init__(self):
        self.gravity = [0, 0, -9.81]
        self.cut_off_precision = 0.001

        self.physics_client_id = 0
        self.plane_id = -1
        self.die_id = -1

        self.start_pos = [0, 0, 5]
        self.start_orientation = bt.getQuaternionFromEuler(np.asarray([0, 180, 0])*deg_to_rad)

    def initialise_simulation(self, run_visible=True):
        if run_visible:
            connect_type = bt.GUI
        else:
            connect_type = bt.DIRECT
        self.physics_client_id = bt.connect(connect_type)
        bt.setAdditionalSearchPath(pybullet_data.getDataPath())
        bt.setGravity(self.gravity[0], self.gravity[1], self.gravity[2], self.physics_client_id)

        # plane_id = bt.createCollisionShape(bt.GEOM_PLANE, planeNormal=[0, 0, 1])
        # self.plane_id = bt.createMultiBody(baseMass=0,
        #                                    baseCollisionShapeIndex=plane_id)
        # bt.changeDynamics(self.plane_id, -1, lateralFriction=0.9, spinningFriction=1, restitution=0.1)
        self.plane_id = bt.loadURDF("plane.urdf")

        # box_id = bt.createCollisionShape(bt.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
        # self.die_id = bt.createMultiBody(baseMass=1,
        #                                  baseCollisionShapeIndex=box_id,
        #                                  basePosition=start_pos,
        #                                  baseOrientation=start_orientation)

        # convex mesh from obj
        die_collision_shape = bt.createCollisionShape(bt.GEOM_MESH, fileName="polygon_exchange.obj")
        self.die_id = bt.createMultiBody(baseMass=1,
                           baseCollisionShapeIndex=die_collision_shape,
                           basePosition=self.start_pos,
                           baseOrientation=self.start_orientation)
        bt.resetBaseVelocity(self.die_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        bt.changeDynamics(self.die_id, -1, lateralFriction=0.9, spinningFriction=1, restitution=0.1)

    def run(self, visible=True):
        self.initialise_simulation(visible)
        for i in range(10000):
            bt.stepSimulation()
            if self.is_settled():
                break
            if visible:
                time.sleep(1. / 240.)
        self.end_pos, self.end_rot = bt.getBasePositionAndOrientation(self.die_id, self.physics_client_id)
        print('position:', self.end_pos, np.asarray(bt.getEulerFromQuaternion(self.end_rot))*rad_to_deg)
        bt.disconnect(self.physics_client_id)

    def is_settled(self, debug=False):
        pos, rot = bt.getBasePositionAndOrientation(self.die_id, self.physics_client_id)
        vel, ang = bt.getBaseVelocity(self.die_id, self.physics_client_id)

        if debug:
            print(pos, rot, vel, ang)

        is_settled = np.linalg.norm(vel) < self.cut_off_precision and \
                     np.linalg.norm(ang) < self.cut_off_precision and \
                     pos[2] < 5

        return is_settled

    def find_up_face_idx(self, normals):
        up_vector = np.asarray(normals[0])
        up_vector_length = np.linalg.norm(up_vector)
        end_rotation = quat.Quaternion(*self.end_rot)

        dots = []
        for normal in normals:
            # rotate normal
            end_normal = end_rotation.rotate(np.asarray(normal))
            end_normal_length = np.linalg.norm(end_normal)

            # compare rotated normal with up_vector
            dot = np.dot(up_vector, end_normal) / (up_vector_length * end_normal_length)
            dots.append(dot)
        print(dots)
        print(np.argmax(np.asarray(dots)), np.max(dots))


if __name__ == '__main__':
    dodeca = create_dodecahedron()
    dodeca.align_normal_to_vector(0, [0, 0, 1])
    dodeca.save_obj(file_name='polygon_exchange')

    roller = DiceRoller()
    roller.run(visible=True)
    roller.find_up_face_idx(dodeca.face_normals)
