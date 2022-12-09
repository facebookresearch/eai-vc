from dotmap import DotMap
import cvxpy as cp
import numpy as np

from cto.constraints import contact_surface, friction_cone
from cto.constraints import sliding_friction


def environment_contact(
    location, orientation, params, type="sticking", sliding_direction=None
):

    # environment contact has fixed location (in the body frame) and can have sliding friction
    constr = []
    force = cp.Variable(3)
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
    contact.cost = 1e-1 * cp.sum_squares(force)

    return contact


def fixed_force_contact(force, surface, params):
    constr = []
    force_skew = cp.Parameter((3, 3), value=skew_symetrify(force))
    n_vertices = len(params.simplices[0])

    if surface == 0:
        in_contact = cp.Parameter(value=0.0)
        simplices = cp.Parameter((n_vertices, 3), value=np.zeros((n_vertices, 3)))
    else:
        in_contact = cp.Parameter(value=1.0)
        simplices = cp.Parameter((n_vertices, 3), value=params.simplices[surface - 1])

    auxiliary_location = cp.Variable(3)
    location = cp.Variable(3)
    constr += [location == in_contact * auxiliary_location]

    location_weights = cp.Variable(n_vertices)  # contact location weights
    torque = -force_skew @ location

    cost = cp.sum_squares(location)
    constr += contact_surface(location, location_weights, simplices, params)

    contact = DotMap()
    contact.type = "fixed_force"
    contact.location = location
    contact.location_weights = location_weights

    contact.force_skew = force_skew
    contact.in_contact = in_contact
    contact.simplices = simplices

    contact.surface = surface
    contact.torque = torque
    contact.constr = constr
    contact.cost = cost

    return contact


def fixed_location_contact(location, surface, params):

    constr = []
    location_skew = cp.Parameter((3, 3), value=skew_symetrify(location))

    if surface == 0:
        in_contact = cp.Parameter(value=0.0)
        orn = cp.Parameter((3, 3), value=np.zeros((3, 3)))
    else:
        in_contact = cp.Parameter(value=1.0)
        orn = cp.Parameter((3, 3), value=params.contact_frame_orientation[surface - 1])

    # contact force in the body frame
    auxiliary_force = cp.Variable(3)
    force = cp.Variable(3)
    constr += [force == in_contact * auxiliary_force]
    torque = location_skew @ force

    cost = cp.sum_squares(force)

    # bounded force
    constr += [force <= params.max_force]
    constr += [force >= -params.max_force]

    # friction cone
    f_local = orn.T @ force
    constr += friction_cone(f_local, params.friction_coeff)

    contact = DotMap()
    contact.type = "fixed_location"
    contact.surface = surface
    contact.orn = orn
    contact.location = location
    contact.location_skew = location_skew
    contact.in_contact = in_contact

    contact.force = force
    contact.surface = surface
    contact.torque = torque

    contact.constr = constr
    contact.cost = cost

    return contact


def skew_symetrify(a):
    # computes a x b with only b being a decision variable
    a1, a2, a3 = a
    A = np.array([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])

    return A
