from gurobipy import GRB
import gurobipy as gp
import cto.utils.gurobi as ugp
from cto.utils.gurobi import MVar
from dotmap import DotMap
import numpy as np
import pinocchio as pin

from cto.trajectory import integrate_trajectory
from cto.miqp.contacts import (
    miqp_contact,
    contact_schedule,
    contact_transition,
    sticking_contact,
)


class MIQP(object):
    def __init__(self, params):
        self.params = params
        self.grb_model = params.grb_model
        self.horizon = params.traj_desired.horizon
        self.n_modes = int(self.horizon / params.contact_duration)

    def setup(self):
        self.contacts = []
        self.constr = []
        self.cost = 0
        self.contact_modes = []
        self.contact_locations = []
        self.location_envelopes = []

        for _ in range(self.n_modes):
            self.contact_modes.append(MVar(self.grb_model, (2, 6), vtype=GRB.BINARY))
            loc = []
            env = []
            for c in range(self.params.n_contacts):
                loc.append(MVar(self.grb_model, (3, 1)))
                env.append(
                    MVar(self.grb_model, (3, self.params.n_envelopes), vtype=GRB.BINARY)
                )
            self.contact_locations.append(loc)
            self.location_envelopes.append(env)

        for n in range(self.horizon):
            # m: contact mode index
            # n: time index along the whole trajectory

            m = int(n / self.params.contact_duration)
            curr_mode = self.contact_modes[m]

            if m >= 1:
                prev_mode = self.contact_modes[m - 1]
                prev_locations = self.contact_locations[m - 1]
            else:
                prev_mode = self.contact_modes[0]
                prev_locations = self.contact_locations[0]

            curr_contacts = []
            for c in range(self.params.n_contacts):
                curr_location = self.contact_locations[m][c]
                curr_envelope = self.location_envelopes[m][c]
                prev_location = prev_locations[c]
                prev_surface = prev_mode[c]
                curr_contact = miqp_contact(curr_location, curr_envelope, self.params)

                curr_contacts.append(curr_contact)
                self.cost += curr_contact.cost
                self.constr += curr_contact.constr
                self.constr += sticking_contact(
                    curr_location, prev_location, prev_surface, self.params
                )

            # contact surface scheduling
            self.constr += contact_transition(curr_mode, prev_mode, m, self.params)
            self.constr += contact_schedule(curr_contacts, curr_mode, self.params)

            # environment contacts
            for ec in self.params.environment_contacts[n]:
                curr_contacts.append(ec)
                self.cost += ec.cost
                self.constr += ec.constr

            self.contacts.append(curr_contacts)

            # newton equation
            total_force = ugp.sum([c.force for c in curr_contacts])
            # implement the constraint as a penalty term significantly speeds up convergence
            # but the solution is not optimal
            force_diff = total_force - self.params.traj_desired.total_force[n]
            self.cost += 1e4 * force_diff.square()
            # self.constr += (total_force == self.params.traj_desired.total_force[n])

            # euler equation
            total_torque = ugp.sum([c.torque for c in curr_contacts])
            self.constr += total_torque == self.params.traj_desired.total_torque[n]

        self.grb_model.setObjective(self.cost, GRB.MINIMIZE)
        _ = self.grb_model.addConstrs(cc for cc in self.constr)

    def solve(self, nsol=1, time_limit=60):
        self.grb_model.setParam("SolutionLimit", nsol)
        self.grb_model.setParam("TimeLimit", time_limit)
        self.grb_model.setParam("MIPFocus", 2)
        self.grb_model.optimize()

        sol = DotMap()
        try:
            sol.forces = self.get_forces()
            sol.locations = self.get_locations()
            sol.approximated_torques = self.get_approximated_torques()
            sol.contact_modes = self.get_contact_modes()
            sol.forces_world = transform_forces_world(sol.forces, self.params)
            sol.locations_world = transform_locations_world(sol.locations, self.params)
            sol.contacts = self.contacts
            return sol
        except:
            return None

    def evaluate_solution(self, sol):
        # integrate solution
        n_segments = self.params.n_desired_poses - 1
        all_pos_err = []
        all_orn_err = []

        for i in range(n_segments):
            idx_start = i * (self.params.contact_duration * self.params.n_modes_segment)
            idx_end = (i + 1) * (
                self.params.contact_duration * self.params.n_modes_segment
            )

            traj_actual = integrate_solution(sol, self.params, idx_start, idx_end)
            pose_end = pin.XYZQUATToSE3(traj_actual.q[-1])
            pose_des = pin.XYZQUATToSE3(self.params.traj_desired.q[idx_end - 1])

            pos_err = np.linalg.norm(pose_end.translation - pose_des.translation)
            orn_err = np.linalg.norm(pin.log3(pose_end.rotation.T @ pose_des.rotation))

            all_pos_err.append(pos_err)
            all_orn_err.append(orn_err)

        err = (100 * np.sum(all_pos_err), 180 / np.pi * np.sum(all_orn_err))
        return err

    def get_forces(self):
        forces = []
        for i in range(self.horizon):
            force = []
            for c in self.contacts[i]:
                force.append(c.force.value)
            forces.append(force)
        return forces

    def get_approximated_torques(self):
        torques = []
        for i in range(self.horizon):
            torque = []
            for c in self.contacts[i]:
                torque.append(c.torque.value)
            torques.append(torque)
        return torques

    def get_locations(self):
        locations = []
        for i in range(self.horizon):
            location = []
            for c in self.contacts[i]:
                if isinstance(c.location, np.ndarray):
                    location.append(c.location)
                else:
                    location.append(c.location.value)
            locations.append(location)
        return locations

    def get_contact_modes(self):
        contact_modes_int = []
        for mode in self.contact_modes:
            # convert one-hot encoding to integer
            mode_int = []
            for c in range(self.params.n_contacts):
                if np.abs(np.sum(mode.value[c])) <= 1e-3:
                    # not in contact
                    mode_int.append(0)
                else:
                    # in contact, note: surface index in the contact mode starts from 1
                    mode_int.append(np.argmax(mode.value[c]) + 1)
            contact_modes_int.append(mode_int)
        return contact_modes_int


def transform_locations_world(locations_body, params):
    q = params.traj_desired.q

    locations_world = []
    for n in range(params.horizon):
        curr_pose = pin.XYZQUATToSE3(q[n])
        p = curr_pose.translation
        R = curr_pose.rotation
        location_world = []
        for c, r in enumerate(locations_body[n]):
            location_world.append(p + R @ r)
        locations_world.append(location_world)

    return locations_world


def transform_forces_world(forces_body, params):
    q = params.traj_desired.q

    forces_world = []
    for n in range(params.horizon):
        curr_pose = pin.XYZQUATToSE3(q[n])
        R = curr_pose.rotation
        force_world = []
        for f in forces_body[n]:
            force_world.append(R @ f)
        forces_world.append(force_world)

    return forces_world


def integrate_solution(sol, params, idx_start=None, idx_end=None):
    if idx_start is None:
        idx_start = 0
    if idx_end is None:
        idx_end = len(params.traj_desired.q)

    traj_length = idx_end - idx_start

    total_force = np.zeros((traj_length, 3))
    total_torque = np.zeros((traj_length, 3))

    pose_start = params.traj_desired.q[idx_start]
    forces = sol.forces[idx_start:idx_end]
    locations = sol.locations[idx_start:idx_end]

    for n in range(traj_length):
        total_force[n] = np.sum(forces[n], axis=0)
        total_torque[n] = np.sum(np.cross(locations[n], forces[n]), axis=0)

    traj = integrate_trajectory(
        pose_start, idx_start, idx_end, total_force, total_torque, params
    )
    return traj
