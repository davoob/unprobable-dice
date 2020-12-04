import pychrono.core as chrono
import pychrono.irrlicht as chronoirr
import numpy as np
from structures.polygon import create_dodecahedron
import random, time


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


dodeca = create_dodecahedron()
# dodeca.extend_side(0, 0.5)
# dodeca.align_normal_to_vector(4, [0, 0, 1])
cut_off = 0.00001
run_visible = True

system = chrono.ChSystemNSC()
system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)

# Contact material for container
ground_mat = chrono.ChMaterialSurfaceNSC()
ground_mat.SetFriction(0.8)

# Shared contact materials for falling objects
dice_mat = chrono.ChMaterialSurfaceNSC()
dice_mat.SetFriction(0.5)

# Create the five walls of the rectangular container, using fixed rigid bodies of 'box' type
floor_body = chrono.ChBodyEasyBox(50, 1, 50, 1000, True, True, ground_mat)
floor_body.SetPos(chrono.ChVectorD(0, 0, 0))
floor_body.SetBodyFixed(True)
system.Add(floor_body)

die_file = dodeca.get_chrono_mesh()
die = chrono.ChBodyEasyMesh(die_file,  # obj filename
                            10000,     # density
                            True,      # compute mass?
                            True,      # visualization?
                            True,      # collision?
                            dice_mat)  # material
system.Add(die)

dice_position = [0, 5, 0]
dice_speed = [0, 0, 0]
dice_rotation = chrono.Q_from_AngAxis(-0 * chrono.CH_C_DEG_TO_RAD, chrono.VECT_Y) * chrono.Q_from_AngAxis(-117 * chrono.CH_C_DEG_TO_RAD, chrono.VECT_Z)
print(dice_rotation, chrono.Q_to_Euler123(dice_rotation) * chrono.CH_C_RAD_TO_DEG)
normals = dodeca.face_normals
for normal in normals:
    ch_normal = chrono.ChVectorD(*normal)
    ch_normal = dice_rotation.Rotate(ch_normal)

    if round(ch_normal ^ chrono.VECT_X, 1) == 1 or round(ch_normal ^ chrono.VECT_Y, 1) == 1 or round(ch_normal ^ chrono.VECT_Z, 1) == 1:
        print(ch_normal)


dice_ang_speed = [0, 0, 0]
# self.dice_position = 10 * ([2, 1, 2] * np.random.random(3) + [-1, 0.5, -1])
# dice_rotation = get_random_rotation()  # 360 * np.random.random(3)
# self.dice_speed = 10 * get_random_vector()
# self.dice_ang_speed = 10 * get_random_vector()

die.SetPos(chrono.ChVectorD(*dice_position))
# rotation = get_rotation_quaternion(*self.dice_rotation)
die.SetRot(dice_rotation)

# die.SetPos_dt(chrono.ChVectorD(*dice_speed))
# die.SetRot_dt(self.dice_ang_speed)
# die.SetWvel_loc(chrono.ChVectorD(*self.dice_ang_speed))

# die.SetPos_dtdt(chrono.ChVectorD(0, 0, 0))
# die.SetRot_dtdt(get_rotation_quaternion(0, 0, 0))

start_t = time.time()
if not run_visible:
    system.SetChTime(0)
    while system.GetChTime() < 100:
        system.DoStepDynamics(0.02)

        # break if velocity and rotational velocity is below threshold
        if die.GetPos_dt().Length() < cut_off and die.GetRot_dt().Length() < cut_off and die.GetPos().y < 2:
            break
else:
    visible_sim = chronoirr.ChIrrApp(system, 'Falling', chronoirr.dimension2du(1024, 768))

    # visible_sim.AddTypicalSky()
    # visible_sim.AddTypicalLogo(chrono.GetChronoDataFile('logo_pychrono_alpha.png'))
    visible_sim.AddTypicalCamera(chronoirr.vector3df(0, 14, -20))
    visible_sim.AddTypicalLights()

    visible_sim.AssetBindAll()
    visible_sim.AssetUpdateAll()
    visible_sim.SetTimestep(0.02)
    visible_sim.SetTryRealtime(True)

    while visible_sim.GetDevice().run():
        visible_sim.BeginScene()
        visible_sim.DrawAll()
        visible_sim.DoStep()
        visible_sim.EndScene()

        # break if velocity and rotational velocity is below threshold
        if die.GetPos_dt().Length() < cut_off and die.GetRot_dt().Length() < cut_off and die.GetPos().y < 2:
            break
end_t = time.time()
duration = end_t - start_t

pos = die.GetPos()
vel = die.GetPos_dt()
rot = die.GetRot()
ang_vel = die.GetWvel_loc()

print(rot, chrono.Q_to_Euler123(rot) * chrono.CH_C_RAD_TO_DEG)
