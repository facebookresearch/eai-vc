import numpy as np
import meshcat.geometry as g
import meshcat.transformations as tf


class Arrow(object):
    def __init__(
        self,
        meshcat_vis,
        name,
        location=[0, 0, 0],
        vector=[0, 0, 1],
        length_scale=1,
        color=0xFF22DD,
    ):
        self.vis = meshcat_vis[name]
        self.cone = self.vis["cone"]
        self.line = self.vis["line"]
        self.material = g.MeshBasicMaterial(color=color, reflectivity=0.5)

        self.location, self.length_scale = location, length_scale
        self.anchor_as_vector(location, vector)

    def _update(self):
        translation = tf.translation_matrix(self.location)
        rotation = self.orientation
        offset = tf.translation_matrix([0, self.length / 2, 0])
        self.pose = translation @ rotation @ offset
        self.vis.set_transform(self.pose)

    def set_length(self, length, update=True):
        self.length = length * self.length_scale
        cone_scale = self.length / 0.08
        self.line.set_object(
            g.Cylinder(height=self.length, radius=0.005), self.material
        )
        self.cone.set_object(
            g.Cylinder(height=0.015, radius=0.01, radiusTop=0.0, radiusBottom=0.01),
            self.material,
        )
        self.cone.set_transform(tf.translation_matrix([0.0, cone_scale * 0.04, 0]))
        if update:
            self._update()

    def set_direction(self, direction, update=True):
        orientation = np.eye(4)
        orientation[:3, 0] = np.cross([1, 0, 0], direction)
        orientation[:3, 1] = direction
        orientation[:3, 2] = np.cross(orientation[:3, 0], orientation[:3, 1])
        self.orientation = orientation
        if update:
            self._update()

    def set_location(self, location, update=True):
        self.location = location
        if update:
            self._update()

    def anchor_as_vector(self, location, vector, update=True):
        self.set_direction(np.array(vector) / np.linalg.norm(vector), False)
        self.set_location(location, False)
        self.set_length(np.linalg.norm(vector), False)
        if update:
            self._update()

    def delete(self):
        self.vis.delete()
