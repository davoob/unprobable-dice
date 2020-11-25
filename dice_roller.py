import pychrono.core as chrono
import pychrono.irrlicht as chronoirr
from polygon import create_dodecahedron
import time


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
        self.container_width = 20
        self.container_length = 20
        self.with_walls = True
        self.wall_height = 50

        self.dice_num = 1
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
                                             100,               # density
                                             True,              # compute mass?
                                             True,              # visualization?
                                             True,              # collision?
                                             self.dice_mat)     # material
            self.system.Add(die_body)
            self.set_start_parameters(die_body)
            dice.append(die_body)
        return dice

    def set_start_parameters(self, die):
        die.SetPos(chrono.ChVectorD(0, 5, 0))
        rotation = get_rotaion_quaternion(10, 39, 28)
        die.SetRot(rotation)

        die.SetPos_dt(chrono.ChVectorD(0, 0, 0))
        die.SetWvel_loc(chrono.ChVectorD(10, 10, 10))

        die.SetPos_dtdt(chrono.ChVectorD(0, 0, 0))
        die.SetRot_dtdt(get_rotaion_quaternion(0, 0, 0))

    def run(self):
        start_t = time.time()
        self.system.SetChTime(0)
        while self.system.GetChTime() < 100:
            self.system.DoStepDynamics(0.02)

            # break if velocity and rotational velocity is below threshold
            if self.dice[0].GetPos_dt().Length() < self.cut_off and self.dice[0].GetRot_dt().Length() < self.cut_off:
                break
        end_t = time.time()
        duration = end_t - start_t
        rot = chrono.Q_to_Euler123(self.dice[0].GetRot()) * chrono.CH_C_RAD_TO_DEG
        pos = self.dice[0].GetPos()

        print(duration, pos, rot)

    def run_multiple(self, num_sim):
        start_t = time.time()
        for i in range(num_sim):
            self.run()
            self.reinitialise_system()
        end_t = time.time()
        duration = end_t - start_t
        print(duration)

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
            if self.dice[0].GetPos_dt().Length() < self.cut_off and self.dice[0].GetRot_dt().Length() < self.cut_off:
                break

        end_t = time.time()
        duration = end_t - start_t
        rot = chrono.Q_to_Euler123(self.dice[0].GetRot()) * chrono.CH_C_RAD_TO_DEG
        pos = self.dice[0].GetPos()

        print(duration, pos, rot)

    def find_up_face_idx(self, normals, die_idx=0):
        up_vector = npvec_to_chvec(normals[0])

        die = self.dice[die_idx]
        rotation = die.GetRot()
        up_vector = rotation.Rotate(up_vector)

        ch_normals = npvecs_to_chvecs(normals)
        response = -1
        for i, ch_normal in enumerate(ch_normals):
            dot = up_vector ^ ch_normal / (up_vector.Length() * ch_normal.Length())
            print(up_vector, ch_normal, dot)
            if round(dot, 4) == 1:
                response = i

        return response

if __name__ == '__main__':
    dodecahedron = create_dodecahedron()
    chrono_dodeca = dodecahedron.get_chrono_mesh()
    test_roller = DiceRoller(chrono_dodeca)
    # test_roller.run()
    # test_roller.reinitialise_system()
    test_roller.run_visible()
    print(test_roller.find_up_face_idx(dodecahedron.face_normals))
