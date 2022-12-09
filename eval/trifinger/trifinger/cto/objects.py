import pinocchio as pin
import numpy as np

from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial import ConvexHull


class Object(object):
    def __init__(self, vertices, urdf_path):
        self.vertices = vertices
        self.convex_hull = ConvexHull(vertices)
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.mass = self.model.inertias[0].mass
        self.inertia = self.model.inertias[0].inertia

    def get_simplices(self, facet_id):
        return self.vertices[self.convex_hull.simplices[facet_id]]

    def get_contact_normal(self, facet_id):
        simplices = self.get_simplices(facet_id)
        edge1 = simplices[0] - simplices[1]
        edge2 = simplices[1] - simplices[2]
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)

        # make sure normal points inward (only works for convex objects)
        # hence, the contact force exerted on the object points along the normal
        normal = np.where(normal @ simplices[0] < 0, normal, -normal)
        return normal

    def get_contact_frame(self, facet_id):
        simplices = self.get_simplices(facet_id)
        y = simplices[0] - simplices[1]
        y = y / np.linalg.norm(y)
        z = self.get_contact_normal(facet_id)
        x = np.cross(y, z)
        return np.vstack((x, y, z)).T


class Cube(Object):
    def __init__(self, length, urdf_path):
        self.width = length
        self.length = length
        self.height = length
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.mass = self.model.inertias[0].mass
        self.inertia = self.model.inertias[0].inertia

    def get_simplices(self, facet_id):
        d = self.length / 2

        # hard-coded surface vertices for a cube
        margin = 0.8
        corners = margin * np.array([[-d, -d], [d, -d], [-d, d], [d, d]])
        simplices = []
        for i in range(3):
            simplices.append(np.insert(corners, i, -d, axis=1))
            simplices.append(np.insert(corners, i, d, axis=1))
        return simplices[facet_id]


class Plate(Object):
    def __init__(self, width, length, height, urdf_path):
        self.width = width / 2
        self.length = length / 2
        self.height = height / 2
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.mass = self.model.inertias[0].mass
        self.inertia = self.model.inertias[0].inertia

    def get_simplices(self, facet_id):
        w = self.width
        l = self.length
        h = self.height

        # hard-coded surface vertices for a plate
        margin = 0.8
        corners1 = margin * np.array([[-w, 0.8 * l], [w, 0.8 * l], [-w, l], [w, l]])
        corners2 = margin * np.array([[-w, -l], [w, -l], [-w, -0.8 * l], [w, -0.8 * l]])
        simplices = []
        simplices.append(np.insert(corners1, 2, -h, axis=1))
        simplices.append(np.insert(corners1, 2, h, axis=1))
        simplices.append(np.insert(corners2, 2, -h, axis=1))
        simplices.append(np.insert(corners2, 2, h, axis=1))
        return simplices[facet_id]
