from dataclasses import dataclass

import numpy as np
import pybullet

"""modified from https://github.com/adamheins/pyb_utils/blob/main/pyb_utils/collision.py"""


@dataclass
class NamedCollisionObject:
    """Name of a body and one of its links.

    The body name must correspond to the key in the `bodies` dict, but is
    otherwise arbitrary. The link name should match the URDF. The link name may
    also be None, in which case the base link (index -1) is used.
    """

    body_name: str
    link_name: str = None


@dataclass
class IndexedCollisionObject:
    """Index of a body and one of its links."""

    body_uid: int
    link_uid: int


def index_collision_pairs(physics_uid, bodies, named_collision_pairs):
    """Convert a list of named collision pairs to indexed collision pairs.

    In other words, convert named bodies and links to the indexes used by
    PyBullet to facilate computing collisions between the objects.

    Parameters:
      physics_uid: Index of the PyBullet physics server to use.
      bodies: dict with body name keys and corresponding indices as values
      named_collision_pairs: a list of 2-tuples of NamedCollisionObject

    Returns: a list of 2-tuples of IndexedCollisionObject
    """

    # build a nested dictionary mapping body names to link names to link
    # indices
    body_link_map = {}
    for name, uid in bodies.items():
        body_link_map[name] = {}
        n = pybullet.getNumJoints(uid, physics_uid)
        for i in range(n):
            info = pybullet.getJointInfo(uid, i, physics_uid)
            link_name = info[12].decode("utf-8")
            body_link_map[name][link_name] = i

    def _index_named_collision_object(obj):
        """Map body and link names to corresponding indices."""
        body_uid = bodies[obj.body_name]
        if obj.link_name is not None:
            link_uid = body_link_map[obj.body_name][obj.link_name]
        else:
            link_uid = -1
        return IndexedCollisionObject(body_uid, link_uid)

    # convert all pairs of named collision objects to indices
    indexed_collision_pairs = []
    for a, b in named_collision_pairs:
        a_indexed = _index_named_collision_object(a)
        b_indexed = _index_named_collision_object(b)
        indexed_collision_pairs.append((a_indexed, b_indexed))

    return indexed_collision_pairs


class CollisionDetector:
    def __init__(self, cid, robot, bodies, named_collision_pairs):
        self.cid = cid
        self.robot = robot  # robot wrapper
        self.indexed_collision_pairs = index_collision_pairs(
            self.cid, bodies, named_collision_pairs
        )

    def compute_distances(self, q, max_distance=1.0):
        """Compute closest distances for a given configuration.

        Parameters:
          q: Iterable representing the desired configuration. This is applied
             directly to PyBullet body with index bodies["robot"].
          max_distance: Bodies farther apart than this distance are not queried
             by PyBullet, the return value for the distance between such bodies
             will be max_distance.

        Returns: A NumPy array of distances, one per pair of collision objects.
        """

        # put the robot in the given configuration
        self.robot.reset_finger_positions_and_velocities(q, np.zeros_like(q))

        # compute shortest distances between all object pairs
        distances = []
        for a, b in self.indexed_collision_pairs:
            closest_points = pybullet.getClosestPoints(
                a.body_uid,
                b.body_uid,
                distance=max_distance,
                linkIndexA=a.link_uid,
                linkIndexB=b.link_uid,
                physicsClientId=self.cid,
            )

            # if bodies are above max_distance apart, nothing is returned, so
            # we just saturate at max_distance. Otherwise, take the minimum
            if len(closest_points) == 0:
                distances.append(max_distance)
            else:
                distances.append(np.min([pt[8] for pt in closest_points]))

        return np.array(distances)

    def in_collision(self, q, margin=0):
        """Returns True if configuration q is in collision, False otherwise.

        Parameters:
          q: Iterable representing the desired configuration.
          margin: Distance at which objects are considered in collision.
             Default is 0.0.
        """
        ds = self.compute_distances(q, max_distance=margin)
        return (ds < margin).any()
