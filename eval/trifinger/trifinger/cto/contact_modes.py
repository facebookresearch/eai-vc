import numpy as np
from itertools import combinations, permutations

# networkx only accepts hashable objects as node labels, so convert tuples -> strings


def label_contact_mode(contact_mode, time_step):
    strings = [str(s) for s in contact_mode]
    label = ",".join(strings) + "," + str(time_step)
    return label


def recover_contact_mode_from_label(label):
    contact_mode_and_time_step = tuple((int(s) for s in label.split(",")))
    return contact_mode_and_time_step[:-1], contact_mode_and_time_step[-1]


def enumerate_contact_modes(params):
    n_contacts = params.n_contacts
    n_surfaces = params.n_surfaces
    allowed_surfaces = set(range(1, n_surfaces + 1)) - params.forbidden_surfaces
    contact_modes = []
    for n_active_contacts in range(n_contacts + 1):
        for surfaces in combinations(allowed_surfaces, n_active_contacts):
            for contacts in permutations(range(n_contacts), n_active_contacts):
                mode = np.zeros(n_contacts).astype(int)
                mode[list(contacts)] = list(surfaces)
                contact_modes.append(list(mode))
    return contact_modes


def find_feasible_transitions(curr_mode, params):
    surfaces = set(range(params.n_surfaces + 1))
    surfaces = surfaces - set(params.forbidden_surfaces)

    def duplicate_surface(mode):
        surface_set = set()
        for surface in mode:
            if surface in surface_set and surface != 0:
                return True
            else:
                surface_set.add(surface)
        return False

    def dfs(c, curr_mode, next_mode, n_diff):
        if c == len(curr_mode):
            if n_diff <= 1 and not duplicate_surface(next_mode):
                next_modes.append(next_mode)
                return
        else:
            if n_diff > 1:
                return
            else:
                if curr_mode[c] == 0:
                    # curr_mode[c] not in contact, s = next_mode[c] can be any surface
                    for s in surfaces:
                        dfs(c + 1, curr_mode, [*next_mode, s], n_diff + int(s != 0))
                else:
                    # curr_mode[c] in contact, s = next_mode[c] can either be 0 or unchanged
                    dfs(c + 1, curr_mode, [*next_mode, 0], n_diff + 1)
                    dfs(c + 1, curr_mode, [*next_mode, curr_mode[c]], n_diff)

    next_modes = []
    dfs(0, curr_mode, [], 0)
    return next_modes


def construct_contact_plan(contact_modes, params):
    d = params.contact_duration
    contact_plan = []
    for i, mode in enumerate(contact_modes):
        for j in range(d):
            contact_plan.append((mode, i * d + j))
    return contact_plan
