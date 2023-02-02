import numpy as np
from dotmap import DotMap

import gurobipy as gp
from gurobipy import GRB

from cto.utils.gurobi import MVar
import cto.utils.gurobi as ugp


def miqp_contact(location, location_envelope, params):
    constr = []
    contact = DotMap()
    m = params.grb_model

    contact.force = contact.f_b = MVar(m, (3, 1))  # contact force in the body frame
    contact.location = contact.r_b = location  # contact location in the body frame
    contact.torque = contact.tau_b = MVar(m, (3, 1))  # contact torque in the body frame
    contact.f_c = MVar(m, (3, 1))  # contact force in the contact frame

    n_simplices = params.simplices[0].shape[0]
    contact.wr_b = MVar(m, (n_simplices, 1))

    contact.location_envelope = location_envelope
    contact.force_envelope = MVar(m, (3, 2), vtype=GRB.BINARY)
    contact.mccormick_envelope = MVar(m, (6, 2 * params.n_envelopes), vtype=GRB.BINARY)

    constr += friction_cone(contact.f_c, params.friction_coeff)
    constr += torque_approximation(contact, params)

    contact.constr = constr
    contact.cost = contact.force.square() + contact.wr_b.square()

    return contact


## friction cone


def friction_cone(f_c, friction_coeff):
    constr = []

    fx, fy, fz = f_c[0, 0], f_c[1, 0], f_c[2, 0]

    # unilateral force
    constr.append(fz >= 0)
    # max. tangential friction
    constr.append(fx <= friction_coeff * fz)
    constr.append(fx >= -friction_coeff * fz)
    constr.append(fy <= friction_coeff * fz)
    constr.append(fy >= -friction_coeff * fz)

    return constr


def constrained_location(weights, simplices, params):
    n_simplices = simplices.shape[0]
    r_constrained = simplices[0] * weights[0, 0]
    for i in range(1, n_simplices):
        r_constrained += simplices[i] * weights[i, 0]
    return MVar(params.grb_model, var=r_constrained.tolist())


def contact_schedule(contacts, contact_mode, params):
    constr = []
    M = params.M

    # each surface can have at most one contact
    constr += contact_mode.sum(axis=0) <= 1
    # each contact can be on at most one surface
    constr += contact_mode.sum(axis=1) <= 1

    for c in range(params.n_contacts):
        for s in range(params.n_surfaces):
            contact = contacts[c]
            f_c = contact.f_c
            f_b = contact.f_b
            r_b = contact.r_b

            # if the contact surface is forbidden,  then the contact is not active
            if s + 1 in params.forbidden_surfaces:
                constr.append(contact_mode[c, s] == 0)

            R = params.contact_frame_orientation[s]

            # force transformation to the contact frame
            constr += f_b - R @ f_c <= M * (1 - contact_mode[c, s])
            constr += R @ f_c - f_b <= M * (1 - contact_mode[c, s])

            # constraining contact location on the contact surface
            constr += contact.wr_b >= 0
            constr += contact.wr_b.sum(axis=0) == 1

            r_constrained = constrained_location(
                contact.wr_b, params.simplices[s], params
            )
            constr += r_b - r_constrained <= M * (1 - contact_mode[c, s])
            constr += r_constrained - r_b <= M * (1 - contact_mode[c, s])

            # contact complementarity
            constr += f_b <= M * contact_mode[c].sum()
            constr += -f_b <= M * contact_mode[c].sum()
            constr += f_c <= M * contact_mode[c].sum()
            constr += -f_c <= M * contact_mode[c].sum()

    return constr


def sticking_contact(curr_location, prev_location, prev_surface, params):
    constr = []
    n_surfaces = params.n_surfaces
    M = params.M

    for s in range(n_surfaces):
        constr += curr_location - prev_location <= M * (1 - prev_surface[s])
        constr += prev_location - curr_location <= M * (1 - prev_surface[s])

    return constr


def contact_transition(curr_mode, prev_mode, mode_idx, params):
    constr = []
    diff = MVar(params.grb_model, shape=curr_mode.shape)
    constr += diff == prev_mode - curr_mode

    # at most 1 contact change each time
    abs_diff = diff.abs()
    constr.append(abs_diff.sum() <= 1)

    # contact cannot be removed if the acc. is not zero
    d = params.contact_duration
    start = mode_idx * d
    acc_norm = np.linalg.norm(params.traj_desired.ddq[start : start + d, :])

    if not np.isclose(acc_norm, 0):
        constr += diff <= 0

    return constr


def environment_contact(
    location, orientation, params, type="sticking", sliding_direction=None
):
    # environment contact has fixed location (in the body frame) and can have sliding friction
    constr = []
    force = MVar(params.grb_model, (3, 1))
    f_local = orientation.T @ force
    if type == "sticking":
        constr += friction_cone(f_local, params.environment_friction)
    elif type == "sliding":
        constr += sliding_friction(
            f_local, sliding_direction, params.environment_friction
        )

    contact = DotMap()
    contact.type = "environment_" + type
    contact.force = force
    contact.location = location
    contact.torque = skew_symetrify(location) @ force
    contact.constr = constr
    contact.cost = 1e-3 * contact.force.square()

    return contact


def sliding_friction(f_local, sliding_direction, friction_coeff):
    constr = []
    sliding_direction = sliding_direction / np.linalg.norm(sliding_direction)

    constr += f_local[:2] == -friction_coeff * f_local[2] * sliding_direction
    constr += f_local[2] >= 0

    return constr


def skew_symetrify(a):
    # computes a x b with only b being a decision variable
    a1, a2, a3 = a
    A = np.array([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])

    return A


def torque_approximation(contact, params):
    constr = []
    r, f, tau = contact.location, contact.force, contact.torque
    zr, zf = contact.location_envelope, contact.force_envelope

    n_envelopes, M = params.n_envelopes, params.M
    lb_r = params.max_location * np.array(
        [-1 + 2 * k / n_envelopes for k in range(n_envelopes)]
    )
    ub_r = params.max_location * np.array(
        [-1 + 2 * (k + 1) / n_envelopes for k in range(n_envelopes)]
    )

    lb_f = params.max_force * np.array([-1 + 2 * k / 2 for k in range(2)])
    ub_f = params.max_force * np.array([-1 + 2 * (k + 1) / 2 for k in range(2)])

    # bilinear terms
    b = MVar(params.grb_model, (6, 1))

    # index set for the bilinear terms in the cross product
    idx_set = [(1, 2), (2, 0), (0, 1), (2, 1), (0, 2), (1, 0)]

    for m in range(6):
        i, j = idx_set[m]
        # b[m] = r[i] * f[j]
        for k in range(n_envelopes):
            for l in range(2):
                z = (zr[i, k] + zf[j, l]) / 2

                constr += -b[m] + r[i] * lb_f[l] + lb_r[k] * f[j] - lb_r[k] * lb_f[
                    l
                ] <= M * (1 - z)
                constr += -b[m] + r[i] * ub_f[l] + ub_r[k] * f[j] - ub_r[k] * ub_f[
                    l
                ] <= M * (1 - z)
                constr += b[m] - r[i] * lb_f[l] - ub_r[k] * f[j] + ub_r[k] * lb_f[
                    l
                ] <= M * (1 - z)
                constr += b[m] - r[i] * ub_f[l] - lb_r[k] * f[j] + lb_r[k] * ub_f[
                    l
                ] <= M * (1 - z)

                constr += r[i] - ub_r[k] <= M * (1 - zr[i, k])
                constr += lb_r[k] - r[i] <= M * (1 - zr[i, k])
                constr += f[j] - ub_f[l] <= M * (1 - zf[j, l])
                constr += lb_f[l] - f[j] <= M * (1 - zf[j, l])

    # one and only one envelope can be selected
    constr += zr.sum(axis=1) == 1
    constr += zf.sum(axis=1) == 1

    # compute the torque from the bilinear terms
    constr += tau == b[:3] - b[3:]

    # store the variables for debugging
    contact.b = b

    return constr


def torque_approximation2(contact, params):
    # this implementation combines location/force envelope indicator in a single binary matrix z
    # note: this matrix is created for each time step even if the location remains the same
    constr = []
    r, f, tau = contact.location, contact.force, contact.torque
    z = contact.mccormick_envelope

    n_envelopes, M = params.n_envelopes, params.M
    lb_r = params.max_location * np.array(
        [-1 + 2 * k / n_envelopes for k in range(n_envelopes)]
    )
    ub_r = params.max_location * np.array(
        [-1 + 2 * (k + 1) / n_envelopes for k in range(n_envelopes)]
    )
    lb_f = params.max_force * np.array([-1 + 2 * k / 2 for k in range(2)])
    ub_f = params.max_force * np.array([-1 + 2 * (k + 1) / 2 for k in range(2)])

    # bilinear terms
    b = MVar(params.grb_model, (6, 1))

    # index set for the bilinear terms in the cross product
    idx_set = [(1, 2), (2, 0), (0, 1), (2, 1), (0, 2), (1, 0)]

    for m in range(6):
        i, j = idx_set[m]
        # b[m] = r[i] * f[j]
        for k in range(n_envelopes):
            for l in range(2):
                env_id = k + l * n_envelopes
                constr += -b[m] + r[i] * lb_f[l] + lb_r[k] * f[j] - lb_r[k] * lb_f[
                    l
                ] <= M * (1 - z[m, env_id])
                constr += -b[m] + r[i] * ub_f[l] + ub_r[k] * f[j] - ub_r[k] * ub_f[
                    l
                ] <= M * (1 - z[m, env_id])
                constr += b[m] - r[i] * lb_f[l] - ub_r[k] * f[j] + ub_r[k] * lb_f[
                    l
                ] <= M * (1 - z[m, env_id])
                constr += b[m] - r[i] * ub_f[l] - lb_r[k] * f[j] + lb_r[k] * ub_f[
                    l
                ] <= M * (1 - z[m, env_id])

                constr += r[i] - ub_r[k] <= M * (1 - z[m, env_id])
                constr += lb_r[k] - r[i] <= M * (1 - z[m, env_id])
                constr += f[j] - ub_f[l] <= M * (1 - z[m, env_id])
                constr += lb_f[l] - f[j] <= M * (1 - z[m, env_id])

    # one and only one envelope can be selected
    constr += z.sum(axis=1) == 1

    # compute the torque from the bilinear terms
    constr += tau == b[:3] - b[3:]

    # store the variables for debugging
    contact.b = b

    return constr
