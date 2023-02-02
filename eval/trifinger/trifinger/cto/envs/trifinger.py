import time
import pathlib
import numpy as np
import pybullet
import pinocchio as pin

from cto.kinematics import inverse_kinematics_3d
from cto.envs.collision import NamedCollisionObject, CollisionDetector
from trifinger_envs.cube_env import SimCubeEnvNYU


class TriFingerAndCube(SimCubeEnvNYU):
    def __init__(
        self,
        params,
        cube_pose=None,
        dt=0.001,
        init_difficulty=-1,
        enable_cameras=False,
        visualization=True,
        finger_type="trifingernyu",
    ):
        super().__init__(
            time_step=dt,
            init_difficulty=init_difficulty,
            enable_cameras=enable_cameras,
            visualization=visualization,
            finger_type=finger_type,
        )
        self.draw_verts = False
        obs = self.reset()
        self.q_default = obs["robot_position"]

        self.finger = self.platform.simfinger
        self.cube = self.platform.cube
        self.finger_id = self.finger.finger_id
        self.cube_id = self.cube._object_id
        self.table_id = self.finger._table_id._object_id
        self.ee_ids = self.finger.kinematics.tip_link_ids

        # reset cube pose if desired
        if cube_pose is not None:
            self.set_cube_pose(cube_pose)

        # update friction coeff
        mu_e = params.environment_friction
        mu_c = params.friction_coeff
        for ee_id in self.ee_ids:
            pybullet.changeDynamics(self.finger_id, ee_id, lateralFriction=mu_c)
        pybullet.changeDynamics(
            self.cube_id,
            -1,
            lateralFriction=mu_c,
            rollingFriction=0.0,
            spinningFriction=0.02,
        )
        pybullet.changeDynamics(
            self.table_id,
            -1,
            lateralFriction=mu_e,
            rollingFriction=0.0,
            spinningFriction=0.02,
        )
        pybullet.setPhysicsEngineParameter(enableConeFriction=1)

        self.create_collision_detector()

    def set_cube_pose(self, pose):
        self.cube.set_state(pose[:3], pose[3:])

    def get_cube_pose(self):
        return self.cube.get_state()

    def get_cube_pose_as_SE3(self):
        pos, quat = self.cube.get_state()
        return pin.XYZQUATToSE3(np.hstack((pos, quat)))

    def add_visual_frame(self, pos, orn, length=0.1, width=1):
        end_point_x = pos + orn @ np.array([length, 0, 0])
        end_point_y = pos + orn @ np.array([0, length, 0])
        end_point_z = pos + orn @ np.array([0, 0, length])

        pybullet.addUserDebugLine(
            pos, end_point_x, lineColorRGB=[1, 0, 0], lineWidth=width
        )
        pybullet.addUserDebugLine(
            pos, end_point_y, lineColorRGB=[0, 1, 0], lineWidth=width
        )
        pybullet.addUserDebugLine(
            pos, end_point_z, lineColorRGB=[0, 0, 1], lineWidth=width
        )

    def start_video_recording(self, file_name):
        """Starts video recording and save as a mp4 file.
        Args:
            file_name (str): The absolute path of the file to be saved.
        """
        self.logging_id = pybullet.startStateLogging(
            pybullet.STATE_LOGGING_VIDEO_MP4, file_name
        )

    def stop_video_recording(self):
        """Stops video recording if any."""
        if hasattr(self, "logging_id"):
            pybullet.stopStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, self.logging_id)

    def create_collision_detector(self):
        collision_bodies = {
            "finger": self.finger_id,
            "cube": self.cube_id,
        }

        cube = NamedCollisionObject("cube")
        self.collision_detectors = []
        for i in range(3):
            deg = i * 120
            collision_links = []
            collision_links.append(
                NamedCollisionObject("finger", "finger_upper_link_{}".format(deg))
            )
            collision_links.append(
                NamedCollisionObject("finger", "finger_middle_link_{}".format(deg))
            )
            collision_links.append(
                NamedCollisionObject("finger", "finger_lower_link_{}".format(deg))
            )
            collision_pairs = [(link, cube) for link in collision_links]
            self.collision_detectors.append(
                CollisionDetector(
                    self.finger._pybullet_client_id,
                    self.finger,
                    collision_bodies,
                    collision_pairs,
                )
            )
