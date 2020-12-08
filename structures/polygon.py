import pickle

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pychrono.core as chrono
import pyquaternion as quat

float_precision = 6


def get_center(points):
    center = np.asarray([0] * len(points[0]), dtype=float)
    for point in points:
        center += point
    center = np.divide(center, len(points))
    return center


def create_dodecahedron():
    h = (np.sqrt(5) - 1) * 0.5
    dodecahedron_points = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1],
                           [-1, -1, -1],
                           [0, (1 + h), (1 - h ** 2)], [0, -(1 + h), (1 - h ** 2)], [0, (1 + h), -(1 - h ** 2)],
                           [0, -(1 + h), -(1 - h ** 2)],
                           [(1 + h), (1 - h ** 2), 0], [-(1 + h), (1 - h ** 2), 0], [(1 + h), -(1 - h ** 2), 0],
                           [-(1 + h), -(1 - h ** 2), 0],
                           [(1 - h ** 2), 0, (1 + h)], [-(1 - h ** 2), 0, (1 + h)], [(1 - h ** 2), 0, -(1 + h)],
                           [-(1 - h ** 2), 0, -(1 + h)]]
    dodecahedron_faces = [[0, 12, 1, 10, 8], [0, 8, 4, 17, 16], [0, 16, 2, 14, 12], [8, 10, 5, 13, 4],
                          [4, 13, 15, 6, 17], [17, 6, 9, 2, 16], [12, 14, 3, 18, 1], [13, 5, 19, 7, 15],
                          [10, 1, 18, 19, 5], [7, 11, 9, 6, 15], [3, 14, 2, 9, 11], [11, 7, 19, 18, 3]]

    dodecahedron = Polygon(dodecahedron_points)
    dodecahedron.add_faces(dodecahedron_faces)
    dodecahedron.set_face_values([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    return dodecahedron


def create_cube():
    cube_points = [[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]]
    cube_faces = [[3, 2, 1, 0], [1, 5, 4, 0], [0, 4, 7, 3], [4, 5, 6, 7], [3, 7, 6, 2], [2, 6, 5, 1]]

    cube = Polygon(cube_points)
    cube.add_faces(cube_faces)
    cube.set_face_values([1, 2, 3, 4, 5, 6])

    return cube


class Polygon:
    def __init__(self, points=None, pickle_file=None):
        self.faces = []
        self.face_points = []
        self.face_center_points = []
        self.face_normals = []
        self.face_values = []

        if points is None:
            self.points = np.asarray([])
            self.center = 0
        else:
            self.points = np.asarray(points).astype(float)
            self.center = get_center(self.points)

        self.number_of_vertices = self.get_number_of_vertices()
        self.number_of_points = len(self.points)

        self.test_angles = []
        self.trapping_ranges = np.zeros(0)
        self.trapping_areas = np.zeros(0)
        self.center_proj = []
        self.trapping_is_anisotropic = False
        self.trapping_is_limited = False
        self.corner_for_angle = []

        self.probabilities = []
        self.mean = 0
        self.std = 0

        if pickle_file is not None:
            with open(pickle_file, 'rb') as file:
                load_instance = pickle.load(file)

            for k in load_instance.__dict__.keys():
                setattr(self, k, getattr(load_instance, k))

    """ add to polygon methods """
    def add_face(self, point_idx, recalculate=True):
        face_points = point_idx
        face_center = get_center(self.points[face_points])
        face = self.get_face_triangles(face_points, len(self.faces))

        self.face_points.append(face_points)
        self.face_center_points.append(face_center)
        self.faces.append(face)

        if recalculate:
            # update number_of_points
            self.number_of_vertices = self.get_number_of_vertices()

            # update face_normals
            self.calculate_face_normals()

            # update centers
            self.recalculate_center()

    def add_faces(self, faces_idx):
        for face_idx in faces_idx:
            self.add_face(face_idx, recalculate=False)

        # update number_of_points
        self.number_of_vertices = self.get_number_of_vertices()

        # update face_normals
        self.calculate_face_normals()

        # update centers
        self.recalculate_center()

    def set_face_values(self, face_values):
        value_num = len(face_values)
        self.face_values = [0] * value_num
        if value_num == len(self.faces) == 1:
            self.face_values[0] = face_values[0]

        elif value_num == len(self.faces) == 6:
            done = []

            self.face_values[0] = face_values[0]
            done.append(0)
            opposite_face = self.get_parallel_face(0)[0]
            self.face_values[opposite_face] = face_values[-1]
            done.append(opposite_face)

            neighboring_faces = self.get_neighbor_face(opposite_face)
            cur_face_value_idx = 1
            for neighboring_face in neighboring_faces:
                # skip if already done
                if neighboring_face in done:
                    continue
                self.face_values[neighboring_face] = face_values[cur_face_value_idx]
                done.append(neighboring_face)
                opposite_face = self.get_parallel_face(neighboring_face)[0]
                self.face_values[opposite_face] = face_values[-(1+cur_face_value_idx)]
                done.append(opposite_face)
                cur_face_value_idx += 1

        elif value_num == len(self.faces) == 12:
            self.face_values[0] = face_values[0]
            opposite_face = self.get_parallel_face(0)[0]
            self.face_values[opposite_face] = face_values[-1]

            neighboring_faces = self.get_neighbor_face(opposite_face)
            cur_face_value_idx = 1
            for neighboring_face in neighboring_faces:
                self.face_values[neighboring_face] = face_values[cur_face_value_idx]
                opposite_face = self.get_parallel_face(neighboring_face)[0]
                self.face_values[opposite_face] = face_values[-cur_face_value_idx-1]
                cur_face_value_idx += 1

        else:
            print('set_face_values for ' + str(value_num) + ' faces not implemented')

    """ manipulate polygon methods """
    def move_point(self, point_idx, move_vector, recalculate=True):
        point = self.points[point_idx]
        point += move_vector

        if recalculate:
            self.recalculate_center()
            self.recalculate_face_centers()

    def move_points(self, points_idx, move_vector, recalculate=True):
        for point_idx in points_idx:
            self.move_point(point_idx, move_vector, recalculate=False)

        if recalculate:
            self.recalculate_center()
            self.recalculate_face_centers()

    def scale_points(self, points_idx, scale_factor, recalculate=True):
        center = get_center(self.points[points_idx])

        for point_idx in points_idx:
            point = self.points[point_idx]
            move_vector = (point - center) * (scale_factor - 1)
            self.move_point(point_idx, move_vector, recalculate=False)

        if recalculate:
            self.recalculate_center()
            self.recalculate_face_centers()

    def scale_face(self, face_idx, scale_factor):
        self.scale_points(self.face_points[face_idx], scale_factor)

    def extend_side(self, face_idx, extend_length):
        points = self.face_points[face_idx]
        for point in points:
            neighbors = self.get_neighbor_points(point)
            free_neighbor = False
            for neighbor in neighbors:
                if neighbor not in points:
                    free_neighbor = neighbor
            assert free_neighbor is not False

            # get move direction
            move_vector = self.points[point] - self.points[free_neighbor]
            # normalize move_vector
            move_vector /= np.linalg.norm(move_vector)
            # set move_vector to extend_length
            move_vector *= extend_length

            self.move_point(point, move_vector, recalculate=False)

        self.recalculate_center()
        self.recalculate_face_centers()

    def rotate(self, angle, axis):
        rotation = quat.Quaternion(axis=axis, angle=angle)
        for i, point in enumerate(self.points):
            point = rotation.rotate(point)
            self.points[i] = point

        for i, point in enumerate(self.face_center_points):
            point = rotation.rotate(point)
            self.face_center_points[i] = point

        self.calculate_face_normals()

    def align_normal_to_vector(self, normal_idx, vector, only_info=False):
        normal = np.asarray(self.face_normals[normal_idx])
        vector = np.asarray(vector)
        vector = vector / np.linalg.norm(vector)

        rot_vector = np.cross(normal, vector)
        rot_vector = rot_vector / np.linalg.norm(rot_vector)

        angle = np.arccos(np.dot(normal, vector))

        if only_info:
            return angle, rot_vector
        self.rotate(angle, rot_vector)

    """ methods to get information """
    def get_neighbor_points(self, point_idx):
        neighbors = []
        for points in self.face_points:
            result = np.where(np.asarray(points) == point_idx)[0]
            if result.size > 0:
                neighbor_1 = points[result[0] - 1]
                if result[0] == len(points) - 1:
                    neighbor_2 = points[0]
                else:
                    neighbor_2 = points[result[0] + 1]

                if neighbor_1 not in neighbors:
                    neighbors.append(neighbor_1)
                if neighbor_2 not in neighbors:
                    neighbors.append(neighbor_2)
        return neighbors

    def get_neighbor_face(self, face_idx):
        points = self.face_points[face_idx]
        neighbor_faces = []
        for idx, face in enumerate(self.face_points):
            if idx == face_idx:
                continue
            for point in points:
                if point in face and idx not in neighbor_faces:
                    neighbor_faces.append(idx)
        return neighbor_faces

    def get_parallel_face(self, face_idx):
        face_normal = self.face_normals[face_idx]
        results = []
        for i, cur_normal in enumerate(self.face_normals):
            if i == face_idx:
                continue
            projection = np.dot(face_normal, cur_normal)
            if np.abs(np.round(projection, float_precision)) == 1:
                results.append(i)

        return results

    def get_trapping_ranges(self):
        return self.trapping_ranges * 180 / np.pi

    def get_probabilities(self):
        values = np.asarray(self.face_values)
        probabilities = np.asarray(self.probabilities)
        sorted_idx = values.argsort()
        sorted_values = values[sorted_idx]
        sorted_probabilities = probabilities[sorted_idx]

        return sorted_values, sorted_probabilities

    def compare_probability_to(self, distribution):
        _, sorted_probabilities = self.get_probabilities()
        num_probabilities = len(sorted_probabilities)
        assert len(distribution) == num_probabilities

        deviations = np.zeros(num_probabilities)
        for i in range(num_probabilities):
            deviations[i] = sorted_probabilities[i] - distribution[i]

        # squared_deviation = np.sum(np.abs(deviations))
        return deviations

    """ internal calculations """
    def recalculate_face_centers(self):
        for i, points in enumerate(self.face_points):
            self.face_center_points[i] = get_center(self.points[points])

    def recalculate_center(self):
        self.center = get_center(self.points)

        self.center_proj = []
        for i, face in enumerate(self.face_points):
            normal = self.face_normals[i]
            center_proj = self.center + normal * np.dot(normal, self.points[face[0]] - self.center)
            self.center_proj.append(center_proj)

    def calculate_face_normals(self):
        self.face_normals = []
        for cur_face in self.face_points:
            cur_normal = np.cross(self.points[cur_face[1]] - self.points[cur_face[0]],
                                  self.points[cur_face[-1]] - self.points[cur_face[0]])
            cur_normal = np.divide(cur_normal, np.linalg.norm(cur_normal))
            self.face_normals.append(cur_normal)

    def get_number_of_vertices(self):
        return len(self.points) + len(self.face_center_points)

    def calculate_isotropic_trapping_angle(self, angle_points=360):
        trapping_angles = []
        for i, face in enumerate(self.face_points):
            normal = self.face_normals[i]
            center_proj = self.center_proj[i]

            face_trapping_angles = []
            for edge_num in range(len(face)):
                second_point_idx = edge_num + 1
                if edge_num == len(face) - 1:
                    second_point_idx = 0

                first_point = self.points[face[edge_num]]
                second_point = self.points[face[second_point_idx]]

                line_normal = np.cross(normal, (second_point - first_point))
                line_normal /= np.linalg.norm(line_normal)

                center_proj_distance = np.dot((center_proj - first_point), line_normal)
                center_proj_to_line_center = -center_proj_distance * line_normal

                center_to_line_center = center_proj + center_proj_to_line_center - self.center

                trapping_angle = np.arcsin(center_proj_distance / np.linalg.norm(center_to_line_center))
                face_trapping_angles.append(trapping_angle)

            trapping_angles.append(min(face_trapping_angles))

        self.test_angles = np.linspace(0, 2 * np.pi, angle_points + 1)
        self.test_angles = self.test_angles[:-1]
        trapping_ranges = []
        for angle in trapping_angles:
            trapping_range = np.asarray([angle]*angle_points)
            trapping_ranges.append(trapping_range)
        self.trapping_ranges = np.asarray(trapping_ranges)
        self.trapping_is_anisotropic = False
        self.trapping_is_limited = False

    def calculate_anisotropic_trapping_angle(self, angle_points=360):
        self.test_angles = np.linspace(0, 2 * np.pi, angle_points + 1)
        self.test_angles = self.test_angles[:-1]

        trapping_ranges = []
        self.corner_for_angle = []
        for i, face in enumerate(self.face_points):
            normal = self.face_normals[i]
            center_proj = self.center_proj[i]

            first_point_vector = self.points[face[0]] - center_proj
            first_point_vector /= np.linalg.norm(first_point_vector)

            cross_product_matrix = np.asarray([[0, -normal[2], normal[1]],
                                               [normal[2], 0, -normal[0]],
                                               [-normal[1], normal[0], 0]])

            face_trapping_range = []
            cur_corner_for_angle = []
            for angle in self.test_angles:
                rotation_matrix = np.identity(3) + np.sin(angle) * cross_product_matrix + (
                        1 - np.cos(angle)) * np.matmul(cross_product_matrix, cross_product_matrix)
                angle_vector = np.matmul(rotation_matrix, first_point_vector)

                center_proj_distances = []
                for corner_idx in face:
                    corner = self.points[corner_idx]
                    center_proj_distance = np.dot(corner - center_proj, angle_vector)
                    center_proj_distances.append(center_proj_distance)
                center_proj_distance = np.max(center_proj_distances)
                relevant_corner = self.points[face[np.argmax(center_proj_distances)]]
                cur_corner_for_angle.append((relevant_corner, center_proj_distance))

                side_vector = center_proj + (center_proj_distance * angle_vector)
                center_side_distance = np.linalg.norm(side_vector - self.center)

                trapping_angle = np.arcsin(center_proj_distance / center_side_distance)
                face_trapping_range.append(trapping_angle)

            self.corner_for_angle.append(cur_corner_for_angle)
            trapping_ranges.append(face_trapping_range)

        self.trapping_ranges = np.asarray(trapping_ranges)
        self.trapping_is_anisotropic = True
        self.trapping_is_limited = False

    def limit_trapping_ranges_opposite(self, override_trapping=False):
        if self.trapping_is_limited or not self.trapping_is_anisotropic:
            print('limit_opposite: trapping is already limited or isotropic')
            if override_trapping:
                self.calculate_anisotropic_trapping_angle()
            else:
                return

        for i, trapping_range in enumerate(self.trapping_ranges):
            num_angles = len(trapping_range)

            assert np.mod(num_angles, 2) == 0

            for idx in range(int(num_angles / 2)):
                opposite_idx = int(np.mod(idx + num_angles / 2, num_angles))
                angle = trapping_range[idx]
                opposite_angle = trapping_range[opposite_idx]
                min_angle = np.min([angle, opposite_angle])

                trapping_range[idx] = min_angle
                trapping_range[opposite_idx] = min_angle

            self.trapping_ranges[i] = trapping_range
        self.trapping_is_limited = True

    def limit_trapping_ranges_impulse(self, override_trapping=False):
        if self.trapping_is_limited or not self.trapping_is_anisotropic:
            print('limit_impulse: trapping is already limited or isotropic')
            if override_trapping:
                self.calculate_anisotropic_trapping_angle()
            else:
                return

        num_angles = len(self.trapping_ranges[0])
        angle_increment_rad = 2 * np.pi / num_angles

        assert np.mod(num_angles, 2) == 0

        for i, trapping_range in enumerate(self.trapping_ranges):
            normal = self.face_normals[i]
            cross_product_matrix = np.asarray([[0, -normal[2], normal[1]],
                                               [normal[2], 0, -normal[0]],
                                               [-normal[1], normal[0], 0]])

            face = self.face_points[i]
            center_proj = self.center_proj[i]

            first_point_vector = self.points[face[0]] - center_proj
            first_point_vector /= np.linalg.norm(first_point_vector)

            trapping_range_limited = []
            for idx, test_angle in enumerate(self.test_angles):
                rotation_matrix = np.identity(3) + np.sin(test_angle) * cross_product_matrix + \
                                  (1 - np.cos(test_angle)) * np.matmul(cross_product_matrix, cross_product_matrix)
                angle_vector = np.matmul(rotation_matrix, first_point_vector)

                relevant_corner = self.corner_for_angle[i][idx][0]

                center_proj_to_corner = relevant_corner - center_proj
                center_proj_to_corner_normed = center_proj_to_corner / np.linalg.norm(center_proj_to_corner)

                gamma = np.arccos(np.round(np.dot(center_proj_to_corner_normed, angle_vector), float_precision))
                idx_range = int(gamma / angle_increment_rad)

                angle_vector_normal = np.cross(normal, angle_vector)
                gamma_direction = int(np.sign(np.dot(center_proj_to_corner_normed, angle_vector_normal)))

                opposite_idx = int(np.mod(idx + num_angles / 2, num_angles))

                trapping_angle = trapping_range[idx]
                trapping_angle_limited = trapping_angle
                for j in range(idx_range + 1):
                    cur_trapping_angle = trapping_range[int(np.mod(opposite_idx + gamma_direction * j, num_angles))]
                    trapping_angle_limited = np.min([trapping_angle_limited, cur_trapping_angle])

                trapping_range_limited.append(trapping_angle_limited)

            self.trapping_ranges[i] = trapping_range_limited
        self.trapping_is_limited = True

    def calculate_trapping_areas(self):
        num_angles = len(self.trapping_ranges[0])
        assert num_angles > 0

        angle_increment_rad = 2 * np.pi / num_angles

        trapping_areas = []
        for i, trapping_range in enumerate(self.trapping_ranges):
            center_to_center_proj_distance = np.linalg.norm(self.center_proj[i] - self.center)

            trapping_area = 0
            for idx in range(num_angles):
                first_idx = idx
                second_idx = np.mod(idx + 1, num_angles)

                first_distance = center_to_center_proj_distance * np.tan(trapping_range[first_idx])
                second_distance = center_to_center_proj_distance * np.tan(trapping_range[second_idx])
                trapping_triangle_area = first_distance * second_distance * np.sin(angle_increment_rad)

                trapping_area += trapping_triangle_area
            trapping_areas.append(trapping_area)

        self.trapping_areas = np.asarray(trapping_areas)

    def calculate_probabilities(self):
        self.probabilities = self.trapping_areas / np.sum(self.trapping_areas)

        self.mean = 0
        for idx in range(len(self.face_values)):
            self.mean += self.probabilities[idx] * self.face_values[idx]
        var = 0
        for idx in range(len(self.face_values)):
            var += (self.face_values[idx] - self.mean)**2 * self.probabilities[idx]
        self.std = np.sqrt(var)

    """ visualization """
    def get_face_triangles(self, face_point_idx, face_center_idx):
        triangles = []
        number_of_points = len(face_point_idx)
        if number_of_points < 3:
            print('not enough points')
            return False
        elif number_of_points == 3:
            triangles = [[face_point_idx]]
        else:
            for i in range(number_of_points):
                if i == number_of_points - 1:
                    triangles.append([face_point_idx[i], face_point_idx[0], self.number_of_points + face_center_idx])
                else:
                    triangles.append(
                        [face_point_idx[i], face_point_idx[i + 1], self.number_of_points + face_center_idx])
        return triangles

    def get_triangles(self):
        faces = []
        for face in self.faces:
            faces += face
        return o3d.utility.Vector3iVector(np.asarray(faces))

    def get_vertices(self):
        if self.number_of_vertices < 3:
            print('not enough points')
            return False
        elif self.number_of_vertices == 3:
            return o3d.utility.Vector3dVector(self.points)
        else:
            return o3d.utility.Vector3dVector(np.concatenate((self.points, np.asarray(self.face_center_points))))

    def get_points(self):
        if self.number_of_vertices < 3:
            print('not enough points')
            return []
        elif self.number_of_vertices == 3:
            return self.points.tolist()
        else:
            return np.concatenate((self.points, np.asarray(self.face_center_points))).tolist()

    def get_trapping_area_mesh(self, area_idx, hover=0.001):
        trapping_range = self.trapping_ranges[area_idx]
        center_proj = self.center_proj[area_idx]

        normal = self.face_normals[area_idx]
        cross_product_matrix = np.asarray([[0, -normal[2], normal[1]],
                                           [normal[2], 0, -normal[0]],
                                           [-normal[1], normal[0], 0]])

        # calculate first_point_vector in for test_angles
        first_point_vector = self.points[self.face_points[area_idx][0]] - center_proj
        first_point_vector /= np.linalg.norm(first_point_vector)
        # calculate side_vector corresponding to first vector
        center_to_center_proj_distance = np.linalg.norm(center_proj - self.center)
        center_proj_to_side_distance = center_to_center_proj_distance * np.tan(trapping_range[0])
        side_vector = center_proj + (center_proj_to_side_distance * first_point_vector)

        trapping_area_points = [center_proj, side_vector]
        trapping_area_triangles = []

        for idx, test_angle in enumerate(self.test_angles):
            rotation_matrix = np.identity(3) + np.sin(test_angle) * cross_product_matrix + \
                              (1 - np.cos(test_angle)) * np.matmul(cross_product_matrix, cross_product_matrix)

            angle_vector = np.matmul(rotation_matrix, first_point_vector)

            # calculating side_vector at next test_angle
            next_point_idx = int(np.mod(idx + 1, len(trapping_range)))
            # skip adding if next_point_idx is 0 (already in trapping_area_points)
            if next_point_idx > 0:
                center_proj_to_side_distance = center_to_center_proj_distance * np.tan(trapping_range[next_point_idx])
                side_vector = center_proj + (center_proj_to_side_distance * angle_vector)
                # and add it to the trapping_area_points
                trapping_area_points.append(side_vector)

            # add triangle with center_proj, current side_vector and side_vector at next test_angle
            # add 1 to idx_s since 0 is taken by center_proj
            trapping_area_triangles.append([0, idx+1, next_point_idx+1])

        # add hovering to mesh
        hover_vector = hover * normal
        trapping_area_points += hover_vector

        area_mesh = o3d.geometry.TriangleMesh()
        area_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(trapping_area_points))
        area_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(trapping_area_triangles))
        return area_mesh

    def get_marking_mesh(self):
        points = []
        faces = []
        for i, value in enumerate(self.face_values):
            if 2 <= value <= 6:
                face = self.face_points[i]
                cur_center_idx = len(points)
                points.append(self.face_center_points[i])
                num_points_to_add = min(value, 5)
                for j in range(num_points_to_add):
                    points.append(self.points[face[j]])

                normal = self.face_normals[i]
                hover_vector = 0.1 * normal
                for j in range(cur_center_idx, cur_center_idx+num_points_to_add+1):
                    points[j] += hover_vector

                for j in range(value-1):
                    if j == 4:
                        faces.append([cur_center_idx, cur_center_idx+j+1, cur_center_idx+1])
                    else:
                        faces.append([cur_center_idx, cur_center_idx+j+1, cur_center_idx+j+2])

        marking_mesh = o3d.geometry.TriangleMesh()
        marking_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(points))
        marking_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces))
        return marking_mesh

    def show(self, show_trapping_areas=False, show_markings=False, save=False, save_name='polygon', save_format='slt'):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = self.get_vertices()
        mesh.triangles = self.get_triangles()

        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        mesh.compute_convex_hull()
        mesh.paint_uniform_color([1, 0.706, 0])

        meshes = [mesh]

        if len(self.trapping_ranges) > 0 and show_trapping_areas:
            for area_idx in range(len(self.trapping_areas)):
                area_mesh = self.get_trapping_area_mesh(area_idx)
                area_mesh.paint_uniform_color([1, 0, 0])
                meshes.append(area_mesh)

        if show_markings:
            marking_mesh = self.get_marking_mesh()
            marking_mesh.paint_uniform_color([0.2, 0.8, 0.2])
            meshes.append(marking_mesh)

        o3d.visualization.draw_geometries(meshes)

        if save:
            try:
                for i, element in enumerate(meshes):
                    full_save_name = save_name + str(i) + '.' + save_format
                    o3d.io.write_triangle_mesh(full_save_name, element)
            except Exception as err:
                print('error during saving polygon' + str(err))

    def get_chrono_mesh(self):
        chrono_mesh = chrono.ChTriangleMeshConnected()
        all_points = self.get_points()
        ch_normals = []

        for i, face in enumerate(self.face_points):
            triangles = self.faces[i]
            for triangle in triangles:
                chrono_mesh.addTriangle(chrono.ChVectorD(*all_points[triangle[0]]),
                                        chrono.ChVectorD(*all_points[triangle[1]]),
                                        chrono.ChVectorD(*all_points[triangle[2]]))
            ch_normal = chrono.ChVectorD(*self.face_normals[i])
            ch_normals.append(ch_normal)

        chrono_mesh.RepairDuplicateVertexes()
        return chrono_mesh

    def save(self, file_name='polygon.pickle'):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # print("Load a ply point cloud, print it, and render it")
    # mesh = o3d.io.read_triangle_mesh("knot.ply")
    #
    # print(mesh)
    # print('Vertices:')
    # print(np.asarray(mesh.vertices))
    # print('Triangles:')
    # print(np.asarray(mesh.triangles))
    #
    #
    # # o3d.visualization.draw_geometries([mesh])
    # # print("A mesh with no normals and no colors does not look good.")
    #
    # print("Computing normal and rendering it.")
    # mesh.compute_vertex_normals()
    #
    # print("Try to render a mesh with normals (exist: " +
    #       str(mesh.has_vertex_normals()) + ") and colors (exist: " +
    #       str(mesh.has_vertex_colors()) + ")")
    #
    # print(np.asarray(mesh.triangle_normals))
    # o3d.visualization.draw_geometries([mesh])

    # dodeca = create_dodecahedron()
    # dodeca.extend_side(0, 0.5)
    # dodeca.extend_side(1, 0.5)
    # dodeca.show()

    # plane_points = [[1, 0, -2], [0, 0, -2], [0, 1, -2],
    #                 [1, 0, 0], [0, 0, 0], [0, 1, 0],
    #                 [1, 0, 1], [0, 0, 1], [0, 1, 1]]
    # plane_faces = [[0, 2, 1], [3, 5, 4], [6, 8, 7]]
    #
    # plane = Polygon(plane_points)
    # plane.add_faces(plane_faces)
    # plane.show()

    # print(dodeca.get_parallel_face(7))
    #
    # dodeca.calculate_anisotropic_trapping_angle(angle_points=720)
    # dodeca.limit_trapping_ranges_opposite()
    # dodeca.calculate_trapping_areas()
    # dodeca.calculate_probabilities()
    # print(dodeca.trapping_areas.tolist())

    # dodeca.show()

    # values, probs = dodeca.get_probabilities()
    # plt.plot(values, probs)
    # plt.show()
    # dodeca.align_normal_to_vector(0, [0, 1, 0])

    # dodeca.show()

    cube = create_cube()
    cube.align_normal_to_vector(0, [0, 1, 1])
    print(cube.face_normals)
    cube.show()
