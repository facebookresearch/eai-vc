import networkx as nx
import numpy as np
import torch
import pinocchio as pin
from torch.utils.data import Dataset
from scipy.special import softmax

from cto.mcts.problems import BiconvexProblem, integrate_solution
from cto.heuristics import kinematic_feasibility, contact_removed, no_contact
from cto.contact_modes import *
from cto.mcts.pvnet import PolicyValueNet, ValueClassifier

eps = 1e-6


class PolicyValueMCTS(object):
    def __init__(self, params, env, networks=None, uct_coeff=0.5, softmax_coeff=0.1):
        self.params = params
        self.env = env
        self.search_tree = nx.DiGraph()
        self.uct_coeff = uct_coeff
        self.softmax_coeff = softmax_coeff
        self.infeasible_contact_plans = set([])

        self.action_ids = {}
        all_contact_modes = enumerate_contact_modes(params)
        for i, mode in enumerate(all_contact_modes):
            self.action_ids[str(mode)] = i
        self.num_actions = len(self.action_ids)
        self.legal_actions = {}

        self.training = False

        self.reached_nodes_reward = {}
        self.reached_nodes_error = {}
        self.reached_nodes_sol = {}
        self.reached_nodes_state = {}

        # create networks if none givne
        if networks is None:
            self.create_networks()
        else:
            self.load_networks(*networks)

        # solve for a dummy state to warmup cvxpy compilation
        dummy_state = [[0] * params.n_contacts] * params.n_contact_modes
        contact_plan = construct_contact_plan(dummy_state, self.params)
        self.biconvex_problem = BiconvexProblem(contact_plan, self.params)
        self.biconvex_problem.setup()
        try:
            _ = self.biconvex_problem.solve()
        except:
            pass

        self.budget = 100
        self.verbose = False

    def is_initial(self, state):
        if len(state) == 1:
            return True
        else:
            return False

    def is_terminal(self, state):
        if len(state) == self.params.n_contact_modes:
            return True
        else:
            return False

    def legal_actions_at(self, state):
        if str(state) in self.legal_actions.keys():
            return self.legal_actions[str(state)]

        if len(state) > 1:
            curr_mode = state[-1]
        else:
            curr_mode = state[0]

        legal_actions = []
        if self.is_terminal(state):
            return legal_actions

        d = self.params.contact_duration
        feasible_transitions = find_feasible_transitions(curr_mode, self.params)
        for next_mode in feasible_transitions:
            n = len(state) * d
            if kinematic_feasibility(next_mode, n, d, self.params, self.env):
                if contact_removed(curr_mode, next_mode):
                    # only allow removing contact for zero acc.
                    acc_norm = np.linalg.norm(
                        self.params.traj_desired.ddq[n : n + d, :]
                    )
                    if np.isclose(acc_norm, 0):
                        legal_actions.append(next_mode)
                else:
                    legal_actions.append(next_mode)

        self.legal_actions[str(state)] = legal_actions
        return legal_actions

    def apply_action_to(self, state, action):
        next_state = [*state, action]
        return next_state

    def add_node(self, state):
        legal_actions = self.legal_actions_at(state)
        p, v = self.predict(state)

        Q = {str(action): 0 for action in legal_actions}
        N = {str(action): 0 for action in legal_actions}
        P = {str(action): p[self.action_ids[str(action)]] for action in legal_actions}

        self.search_tree.add_node(str(state), state=state, Q=Q, N=N, P=P)
        return v

    def add_edge(self, state, next_state, action):
        self.search_tree.add_edge(str(state), str(next_state), action=action)

    def train(self, state, budget=100, max_it=100000, verbose=False):
        # this does not train the NNs
        self.training = True
        self.budget = budget
        self.verbose = verbose

        for i in range(max_it):
            if self.budget <= 0:
                if self.verbose:
                    print("# found solutions:", len(self.reached_nodes_reward))
                return
            else:
                v = self.simulate(state)

    def run(self, state, budget=100, n_candidates=1, max_it=100000, verbose=False):
        self.training = False
        self.verbose = verbose
        self.budget = budget

        for i in range(max_it):
            if self.budget <= 0:
                return
            if len(self.reached_nodes_reward) >= n_candidates:
                return
            v = self.simulate(state)

    def get_solution(self):
        if len(self.reached_nodes_reward) > 0:
            best_node = max(
                self.reached_nodes_reward, key=self.reached_nodes_reward.get
            )
            sol = self.reached_nodes_sol[best_node]
            state = self.reached_nodes_state[best_node]
            return state, sol
        else:
            return None, None

    def act(self, state):
        legal_actions = self.legal_actions_at(state)
        if len(legal_actions) == 0:
            return None

        U = np.array([self.uct(state, action) for action in legal_actions])
        if self.training:
            # softmax
            prob = softmax(U / self.softmax_coeff)
            chosen_idx = np.random.choice(len(legal_actions), p=prob)
        else:
            # max with random tie breaking
            chosen_idx = np.random.choice(np.where(U == U.max())[0])

        action = legal_actions[chosen_idx]

        return action

    def simulate(self, state):

        if self.is_terminal(state):
            return self.compute_reward(state)

        if str(state) not in self.search_tree.nodes():
            v1 = self.add_node(state)
            v2 = self.heuristic_simulate(state)
            return np.max((v1, v2))

        else:
            action = self.act(state)
            if action is None:
                return 0

            next_state = self.apply_action_to(state, action)
            v = self.simulate(next_state)
            N, Q, _ = self.get_state_info(state)

            Qsa = Q[str(action)]
            Nsa = N[str(action)]

            Q[str(action)] = (Nsa * Qsa + v) / (Nsa + 1)
            N[str(action)] += 1

            return v

    def heuristic_simulate(self, state):
        while True:
            if self.is_terminal(state):
                reward = self.compute_reward(state)
                break

            else:
                legal_actions = self.legal_actions_at(state)
                if len(legal_actions) != 0:
                    next_states = [
                        self.apply_action_to(state, action) for action in legal_actions
                    ]
                    heuristics_score = np.array(
                        [self.compute_heuristics_score(state) for state in next_states]
                    )

                    prob = softmax(heuristics_score / self.softmax_coeff)
                    random_idx = np.random.choice(len(next_states), p=prob)
                    state = next_states[random_idx]
                else:
                    return 0

        return reward

    def compute_heuristics_score(self, state):
        _, v = self.predict(state)
        return v

    def get_state_info(self, state):

        N = self.search_tree.nodes[str(state)]["N"]
        Q = self.search_tree.nodes[str(state)]["Q"]
        P = self.search_tree.nodes[str(state)]["P"]

        return N, Q, P

    def load_networks(self, pvnet, value_classifier):
        self.pvnet = pvnet
        self.value_classifier = value_classifier

    def create_networks(self, device="cpu"):
        input_size = self.params.n_contacts
        action_size = self.num_actions
        self.pvnet = PolicyValueNet(input_size, action_size).to(device)
        self.value_classifier = ValueClassifier(input_size, action_size).to(device)

    def load_pvnet(self, model_path, device="cpu"):
        self.pvnet.load_state_dict(torch.load(model_path, map_location=device))

    def load_value_classifier(self, model_path, device="cpu"):
        self.value_classifier.load_state_dict(
            torch.load(model_path, map_location=device)
        )

    def predict(self, state):
        self.pvnet.eval()
        self.value_classifier.eval()
        with torch.no_grad():
            goal = self.get_state_goal(state)
            state = torch.tensor(state).float()[None, :, :]
            goal = torch.tensor(goal).float()[None, :]
            y = self.value_classifier(state, goal)

            if (
                torch.sigmoid(y) > self.params.classifier_threshold
            ):  # tune this threshold on the test set
                p_pred, v_pred = self.pvnet(state, goal)
                p_pred = np.exp(p_pred.detach().cpu().numpy().squeeze())
                v_pred = v_pred.detach().cpu().numpy().item()
            else:
                p_pred = np.ones(self.num_actions) / self.num_actions
                v_pred = 0
        return p_pred, v_pred

    def get_state_goal(self, state):
        d = self.params.contact_duration
        future_horizon = self.params.n_modes_segment * d
        idx_prev = (len(state) - 1) * d
        idx_future = np.min((idx_prev + future_horizon - 1, self.params.horizon - 1))
        pose_prev = pin.XYZQUATToSE3(self.params.traj_desired.q[idx_prev, :])
        pose_future = pin.XYZQUATToSE3(self.params.traj_desired.q[idx_future, :])
        pose_diff = pin.log6(pose_prev.actInv(pose_future)).vector
        goal = np.hstack(
            (pose_prev.translation, pin.log3(pose_prev.rotation), pose_diff)
        )
        return goal

    def get_data(self):
        states = []
        values = []
        action_probs = []
        goals = []

        for node in self.search_tree.nodes():
            state = self.search_tree.nodes[node]["state"]
            N, Q, _ = self.get_state_info(state)
            goal = self.get_state_goal(state)

            action_prob = np.zeros(len(self.action_ids))
            action_value = np.zeros(len(self.action_ids))

            for action, na in N.items():
                action_id = self.action_ids[action]
                action_prob[action_id] = na
                action_value[action_id] = Q[action]

            action_prob = action_prob / (np.sum(action_prob) + eps)
            action_value = (
                action_prob * action_value
            )  # weight action value by its frequency
            value = sum(action_value)

            action_probs.append(action_prob)
            states.append(state)
            values.append(value)
            goals.append(goal)

        return states, values, action_probs, goals

    def uct(self, state, action, eps=1e-6):

        N, Q, P = self.get_state_info(state)
        Nsa = N[str(action)]
        Nsum = sum(N.values())
        exploitation_score = Q[str(action)]
        exploration_score = (
            self.uct_coeff * P[str(action)] * np.sqrt(1 + Nsum) / (1 + Nsa)
        )
        U = exploitation_score + exploration_score

        return U

    def compute_reward(self, state):

        if str(state) in self.reached_nodes_reward.keys():
            reward = self.reached_nodes_reward[str(state)]
            if self.verbose:
                print("reached a previously found solution with reward:", reward)

        elif str(state) in self.infeasible_contact_plans:
            reward = 0

        else:
            self.budget -= 1
            contact_plan = construct_contact_plan(state, self.params)
            if self.verbose:
                print("try contact plan:", state, "budget left:", self.budget)
            try:
                self.biconvex_problem.update(contact_plan)
                sol = self.biconvex_problem.solve()
                reward = self.evaluate_solution(state, sol)

            except:
                # this contact plan does not produce feasible solution
                self.infeasible_contact_plans.add(str(state))
                reward = 0

        return reward

    def evaluate_solution(self, state, sol):
        err_threshold = self.params.err_threshold

        # integrate solution
        reward = 0
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

            err = pos_err + 0.1 * orn_err

            all_pos_err.append(pos_err)
            all_orn_err.append(orn_err)

            if err <= err_threshold:
                reward += (np.exp(err_threshold - err) - 1) / (
                    np.exp(err_threshold) - 1
                )  # normalized to 0-1
            else:
                # if any segment is infeasible the plan is infeasible
                self.infeasible_contact_plans.add(str(state))
                return 0

        reward = reward / n_segments

        if reward > 0:
            self.reached_nodes_sol[str(state)] = sol
            self.reached_nodes_state[str(state)] = state
            self.reached_nodes_reward[str(state)] = reward
            self.reached_nodes_error[str(state)] = (
                100 * np.sum(all_pos_err),
                180 / np.pi * np.sum(all_orn_err),
            )
            if self.verbose:
                print("found a solution with reward:", reward)
        return reward


class MCTSDataset(Dataset):
    def __init__(self, states, values, action_probs, goals):
        super().__init__()
        self.states = states
        self.values = values
        self.action_probs = action_probs
        self.goals = goals

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.values[idx],
            self.action_probs[idx],
            self.goals[idx],
        )
