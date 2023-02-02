import time
import pathlib
import numpy as np
import pybullet
import pinocchio as pin

from bullet_utils.env import BulletEnv
from cto.kinematics import inverse_kinematics_3d
from robot_properties_nyu_finger.config import (
    NYUFingerDoubleConfig0,
    NYUFingerDoubleConfig1,
)
from robot_properties_nyu_finger.wrapper import NYUFingerRobot
from cto.envs.collision import NamedCollisionObject, CollisionDetector


class FingerDoubleAndObject(BulletEnv):
    resources_path = pathlib.Path(__file__).parent.absolute() / "resources"

    def __init__(
        self,
        params,
        box_pos,
        box_orn,
        ee_init=None,
        server=pybullet.GUI,
        dt=0.001,
        box_type="box",
    ):
        super().__init__(server, dt)

        # friction coeff
        mu_e = params.environment_friction
        mu_c = params.friction_coeff
        pybullet.setPhysicsEngineParameter(enableConeFriction=1)

        # Zoom onto the robot
        pybullet.resetDebugVisualizerCamera(0.65, 30, -45, (0, 0.0, 0.0))

        # Disable the gui controller as we don't use them.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

        # Create a robot instance. This adds the robot to the simulator as well.
        config0 = NYUFingerDoubleConfig0()
        config1 = NYUFingerDoubleConfig1()

        self.finger0 = NYUFingerRobot(config=config0)
        self.finger1 = NYUFingerRobot(config=config1)
        self.ee0_id = self.finger0.end_eff_ids[0]
        self.ee1_id = self.finger1.end_eff_ids[0]

        self.ee_ids = [self.ee0_id, self.ee1_id]
        self.fingers = [self.finger0, self.finger1]
        self.configs = [config0, config1]

        # Add the robot to the env
        self.add_robot(self.finger0)
        self.add_robot(self.finger1)
        pybullet.changeDynamics(
            self.finger0.robot_id,
            self.finger0.bullet_endeff_ids[0],
            lateralFriction=mu_c,
        )
        pybullet.changeDynamics(
            self.finger1.robot_id,
            self.finger1.bullet_endeff_ids[0],
            lateralFriction=mu_c,
        )

        # Reset the robot to some initial state.
        dq = np.zeros(self.finger0.nv)
        q = np.array([0.0, np.pi / 3, -(np.pi / 2 + np.pi / 3)])
        self.q_default = q

        if ee_init is None:
            self.finger0.reset_state(q, dq)
            self.finger1.reset_state(q, dq)
        else:
            ee0_init, ee1_init = ee_init
            q0, _ = inverse_kinematics_3d(
                self.finger0.pin_robot, ee0_init, self.ee0_id, q_null=q
            )
            q1, _ = inverse_kinematics_3d(
                self.finger1.pin_robot, ee1_init, self.ee1_id, q_null=q
            )
            self.finger0.reset_state(q0, dq)
            self.finger1.reset_state(q1, dq)

        # add a plane
        plane_urdf = str(self.resources_path / "table.urdf")
        self.plane_id = pybullet.loadURDF(plane_urdf)
        pybullet.resetBasePositionAndOrientation(
            self.plane_id, [0.0, 0.0, params.ground_height], (0.0, 0.0, 1.0, 1.0)
        )
        pybullet.changeDynamics(self.plane_id, -1, lateralFriction=mu_e)

        # add an object
        box_urdf = params.object_urdf
        print(box_urdf)
        self.box_id = pybullet.loadURDF(box_urdf)
        pybullet.resetBasePositionAndOrientation(self.box_id, box_pos, box_orn)
        pybullet.changeDynamics(
            self.box_id,
            -1,
            lateralFriction=mu_c,
            rollingFriction=0.0,
            spinningFriction=0.02,
        )
        pybullet.changeVisualShape(self.box_id, -1, rgbaColor=[0, 0, 0, 0.5])

        self.create_collision_detector()

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

    def step(self, sleep_factor=0):
        """Integrates the simulation one step forward.
        Args:
            sleep (bool, optional): Determines if the simulation sleeps for :py:attr:`~dt` seconds at each step. Defaults to False.
        """
        time.sleep(sleep_factor * self.dt)
        pybullet.stepSimulation()

        for robot in self.robots:
            robot.compute_numerical_quantities(self.dt)

    def close(self):
        pybullet.disconnect(self.physics_client)

    def set_box_pose(self, pose):
        box_pos = pose[:3]
        box_orn = pose[3:]
        pybullet.resetBasePositionAndOrientation(self.box_id, box_pos, box_orn)

    def get_box_pose(self):
        pos, orn = pybullet.getBasePositionAndOrientation(self.box_id)
        pose = pin.XYZQUATToSE3(pos + orn)
        return pose

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
        collision_bodies0 = {
            "robot": self.finger0.robot_id,
            "box": self.box_id,
        }
        collision_bodies1 = {
            "robot": self.finger1.robot_id,
            "box": self.box_id,
        }

        box = NamedCollisionObject("box")

        finger0_upper_link = NamedCollisionObject("robot", "finger0_upper_link")
        finger0_middle_link = NamedCollisionObject("robot", "finger0_middle_link")
        finger0_lower_link = NamedCollisionObject("robot", "finger0_lower_link")

        col_detector0 = CollisionDetector(
            self.physics_client,
            self.finger0,
            collision_bodies0,
            [
                (finger0_upper_link, box),
                (finger0_middle_link, box),
                (finger0_lower_link, box),
            ],
        )

        finger1_upper_link = NamedCollisionObject("robot", "finger1_upper_link")
        finger1_middle_link = NamedCollisionObject("robot", "finger1_middle_link")
        finger1_lower_link = NamedCollisionObject("robot", "finger1_lower_link")

        col_detector1 = CollisionDetector(
            self.physics_client,
            self.finger1,
            collision_bodies1,
            [
                (finger1_upper_link, box),
                (finger1_middle_link, box),
                (finger1_lower_link, box),
            ],
        )

        self.col_detectors = [col_detector0, col_detector1]
