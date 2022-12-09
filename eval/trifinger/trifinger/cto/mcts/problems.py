from re import M
import cvxpy as cp
from dotmap import DotMap
import numpy as np
import pinocchio as pin

from cto.mcts.contacts import (
    skew_symetrify,
    fixed_force_contact,
    fixed_location_contact,
)
from cto.trajectory import integrate_trajectory


class ForceProblem(object):
    def __init__(self, locations, contact_plan, params):
        self.horizon = len(contact_plan)
        self.contact_plan = contact_plan
        self.params = params
        self.locations = locations

    def setup(self):
        self.contacts = []
        self.constr = []
        self.costs = []

        for i in range(self.horizon):
            # i: index along the given contact_plan
            # n: time index along the whole trajectory
            curr_mode, n = self.contact_plan[i]
            curr_contacts = []

            for c, s in enumerate(curr_mode):
                flc = fixed_location_contact(self.locations[i][c], s, self.params)
                curr_contacts.append(flc)
                self.costs.append(flc.cost)
                self.constr += flc.constr

            # environment contacts
            for ec in self.params.environment_contacts[n]:
                curr_contacts.append(ec)
                self.costs.append(ec.cost)
                self.constr += ec.constr

            self.contacts.append(curr_contacts)

            # newton equation
            total_force = cp.sum([c.force for c in curr_contacts])
            self.costs.append(
                1e4
                * cp.sum_squares(total_force - self.params.traj_desired.total_force[n])
            )
            # euler equation
            total_torque = cp.sum([c.torque for c in curr_contacts])
            self.costs.append(
                1e6
                * cp.sum_squares(
                    total_torque - self.params.traj_desired.total_torque[n]
                )
            )

        self.objective = cp.Minimize(cp.sum(self.costs))
        self.prob = cp.Problem(self.objective, self.constr)

    def update(self, contact_plan, locations):
        self.contact_plan = contact_plan
        self.locations = locations

        for i in range(self.horizon):
            # i: index along the given contact_plan
            # n: time index along the whole trajectory
            curr_mode, n = self.contact_plan[i]

            for c, s in enumerate(curr_mode):
                flc = self.contacts[i][c]
                flc.location = self.locations[i][c]
                flc.location_skew.value = skew_symetrify(flc.location)
                flc.surface = s

                if s == 0:
                    flc.in_contact.value = 0.0
                    flc.orn.value = np.zeros((3, 3))
                else:
                    flc.in_contact.value = 1.0
                    flc.orn.value = self.params.contact_frame_orientation[s - 1]

    def solve(self, verbose=False):
        sol = self.prob.solve(warm_start=False, verbose=verbose)

        return sol, self.prob

    def get_forces(self):
        forces = []
        for i in range(self.horizon):
            force = []
            for c in self.contacts[i]:
                force.append(c.force.value)
            forces.append(force)
        return forces


class LocationProblem(object):
    def __init__(self, forces, contact_plan, params):
        self.horizon = len(contact_plan)
        self.contact_plan = contact_plan
        self.params = params
        self.forces = forces
        self.sticking = cp.Parameter(
            (self.horizon, self.params.n_contacts),
            value=np.zeros((self.horizon, self.params.n_contacts)),
        )
        self.environment_torque = cp.Parameter(
            (self.horizon, 3), value=np.zeros((self.horizon, 3))
        )

    def set_sticking_contact_flags(self):
        # setup sticking contact parameters
        for i in range(self.horizon):
            # i: index along the given contact plan
            # n: time index along the whole trajectory
            curr_mode, _ = self.contact_plan[i]
            if i >= 1:
                prev_mode, _ = self.contact_plan[i - 1]
                for c, s in enumerate(curr_mode):
                    if s == prev_mode[c] and s != 0:
                        self.sticking.value[i, c] = 1
                    else:
                        self.sticking.value[i, c] = 0

    def set_environment_torque(self):
        for i in range(self.horizon):
            _, n = self.contact_plan[i]
            self.environment_torque.value[i] = np.sum(
                [ec.torque.value for ec in self.params.environment_contacts[n]], axis=0
            )

    def setup(self):
        self.contacts = []
        self.constr = []
        self.costs = []

        for i in range(self.horizon):
            # i: index along the given contact plan
            # n: time index along the whole trajectory
            curr_mode, n = self.contact_plan[i]
            curr_contacts = []

            for c, s in enumerate(curr_mode):
                ffc = fixed_force_contact(self.forces[i][c], s, self.params)
                curr_contacts.append(ffc)
                self.costs.append(ffc.cost)
                self.constr += ffc.constr

                # sticking contact constraints
                if i >= 1:
                    prev_ffc = self.contacts[i - 1][c]
                    self.constr.append(
                        self.sticking[i, c]
                        * (ffc.location_weights - prev_ffc.location_weights)
                        == 0
                    )

            # environment contact locations are fixed, no need to add into contact list
            self.contacts.append(curr_contacts)

            # euler equation
            total_torque = (
                cp.sum([ffc.torque for ffc in curr_contacts])
                + self.environment_torque[i]
            )
            self.costs.append(
                1e6
                * cp.sum_squares(
                    total_torque - self.params.traj_desired.total_torque[n]
                )
            )

        self.objective = cp.Minimize(cp.sum(self.costs))
        self.prob = cp.Problem(self.objective, self.constr)

    def update(self, contact_plan, forces):
        self.contact_plan = contact_plan
        self.forces = forces

        self.set_sticking_contact_flags()
        self.set_environment_torque()
        n_vertices = len(self.params.simplices[0])

        for i in range(self.horizon):
            # i: index along the given contact_plan
            # n: time index along the whole trajectory
            curr_mode, n = self.contact_plan[i]

            for c, s in enumerate(curr_mode):
                ffc = self.contacts[i][c]
                ffc.force = self.forces[i][c]
                ffc.force_skew.value = skew_symetrify(ffc.force)
                ffc.surface = s

                if s == 0:
                    ffc.in_contact.value = 0.0
                    ffc.simplices.value = np.zeros((n_vertices, 3))
                else:
                    ffc.in_contact.value = 1.0
                    ffc.simplices.value = self.params.simplices[s - 1]

    def solve(self, verbose=False):
        sol = self.prob.solve(warm_start=False, verbose=verbose)

        return sol, self.prob

    def get_locations(self):
        locations = []

        for i in range(self.horizon):
            location = []
            for c in self.contacts[i]:
                if isinstance(c.location, np.ndarray):
                    location.append(c.location)
                else:
                    location.append(c.location.value)
            location += [ec.location for ec in self.params.environment_contacts[i]]
            locations.append(location)
        return locations


class BiconvexProblem(object):
    def __init__(self, contact_plan, params, max_it=1):
        self.contact_plan = contact_plan
        self.params = params
        self.locations = initialize_locations(self.contact_plan, self.params)
        self.forces = initialize_forces(self.contact_plan, self.params)
        self.max_it = max_it

        self.fp = ForceProblem(self.locations, self.contact_plan, self.params)
        self.lp = LocationProblem(self.forces, self.contact_plan, self.params)

    def setup(self):
        self.fp.setup()
        self.lp.setup()

    def update(self, contact_plan):
        self.contact_plan = contact_plan
        self.locations = initialize_locations(self.contact_plan, self.params)
        self.forces = initialize_forces(self.contact_plan, self.params)
        self.fp.update(contact_plan, self.locations)
        self.lp.update(contact_plan, self.forces)

    def solve(self, verbose=False):
        for i in range(self.max_it):
            cost, _ = self.fp.solve()
            self.forces = self.fp.get_forces()
            self.lp.update(self.contact_plan, self.forces)
            cost, _ = self.lp.solve()
            self.locations = self.lp.get_locations()

        sol = DotMap()
        sol.cost = cost
        sol.forces = self.forces
        sol.locations = self.locations
        sol.forces_world = transform_forces_world(self.forces, self.params)
        sol.locations_world = transform_locations_world(self.locations, self.params)
        sol.contacts = self.lp.contacts
        return sol


def initialize_locations(contact_plan, params):
    horizon = params.horizon
    locations = []

    for i in range(horizon):
        location = []
        mode = contact_plan[i][0]
        for s in mode:
            if s == 0:
                location.append(np.zeros(3))
            else:
                simplices = params.simplices[s - 1]
                location.append(np.mean(simplices, axis=0))

        location += [ec.location for ec in params.environment_contacts[i]]
        locations.append(location)
    return locations


def initialize_forces(contact_plan, params):
    horizon = params.horizon
    forces = []

    for i in range(horizon):
        force = []
        mode = contact_plan[i][0]
        for _ in mode:
            force.append(np.zeros(3))

        force += [np.zeros(3) for _ in params.environment_contacts[i]]
        forces.append(force)
    return forces


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
