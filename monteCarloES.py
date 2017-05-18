# Monte Carlo with Exploring Starts

import invertedPendulumGame as ipg
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

WINDOWDIMS = (1200, 600)
CARTDIMS = (50, 10)
PENDULUMDIMS = (6, 200)
GRAVITY = 0.13
REFRESHFREQ = 100
A_CART = 0.15

class MCESagent(object):
    def __init__(self, pendulum, max_episode_length):
        """accepts bin divisions to create discretized states
        x_bins and theta_bins given in range 0 to 1
        v_bins and omega_bins given in actual values
        """
        assert isinstance(pendulum, ipg.InvertedPendulum)
        self.pendulum = pendulum
        self.max_episode_length = max_episode_length
        self.WINDOWWIDTH = self.pendulum.WINDOWWIDTH
        self.epsilon = 10 ** (-10)
        self.x_bins = [self.pendulum.CARTWIDTH / 2,
                       self.WINDOWWIDTH - self.pendulum.CARTWIDTH / 2 - self.epsilon,
                       np.inf]
        self.coarse_xs = [-1, 0, 1]
        self.theta_bins = [-np.pi / 2, -0.1, 0.1,
                           np.pi / 2 - self.epsilon, np.inf]
        self.coarse_thetas = [-2, -1, 0, 1, 2]
        self.v_bins = [-2 * self.pendulum.A_CART,
                       2 * self.pendulum.A_CART, np.inf]
        self.coarse_vs = [-1, 0, 1]
        self.omega_bins = [-0.05, 0.05, np.inf]
        self.coarse_omegas = [-1, 0, 1]
        self.policy = dict()
        self.returns = defaultdict(list)
        self.Q = defaultdict(float)

    @staticmethod
    def fine_to_coarse(value, bins, labels):
        for i, b in enumerate(bins):
            if value <= b:
                return labels[i]

    def coarse_state(self):
        is_dead, t, x, v, theta, omega = self.pendulum.get_state()
        x_coarse = self.fine_to_coarse(x, self.x_bins, self.coarse_xs)
        v_coarse = self.fine_to_coarse(v, self.v_bins, self.coarse_vs)
        theta_coarse = self.fine_to_coarse(theta, self.theta_bins,
                                           self.coarse_thetas)
        omega_coarse = self.fine_to_coarse(omega, self.omega_bins,
                                           self.coarse_omegas)
        return is_dead, (x_coarse, v_coarse, theta_coarse, omega_coarse)

    def actions(self, previous_action, coarse_state):
        """defines available actions given state and previous action
        must stop moving in one direction before moving in the other
        """
        x, _, _, _ = coarse_state
        if previous_action == 'None':
            if x == self.coarse_xs[0]:
                return ['None', 'Right']
            elif x == self.coarse_xs[-1]:
                return ['Left', 'None']
            else:
                return ["Left", "None", "Right"]
        elif previous_action == 'Left':
            if x == self.coarse_xs[0]:
                return ['None']
            else:
                return ['Left', 'None']
        elif previous_action == 'Right':
            if x == self.coarse_xs[-1]:
                return ['None']
            else:
                return ['None', 'Right']

    def episode(self):
        x = np.random.uniform(self.pendulum.CARTWIDTH / 2,
                                    self.WINDOWWIDTH - self.pendulum.CARTWIDTH / 2)
        theta = np.random.uniform(-np.pi / 2, np.pi / 2)
        v = np.random.normal(0, 1.0)
        omega = np.random.normal(0, 0.1)
        is_dead = False
        self.pendulum.set_state((is_dead, 0, x, v, theta, omega))
        is_dead, state = self.coarse_state()
        state_action = []
        previous_action = 'None'
        i = 0
        while not is_dead and i < self.max_episode_length:
            i += 1
            if state in self.policy:
                action = self.policy[state]
            else:
                action = 'None'
            self.pendulum.update_state(action)
            is_dead, state = self.coarse_state()
            state_action.append((previous_action, state, action))
            previous_action = action
        return state_action

    def update_returns(self, state_action):
        """ updates self.returns AND self.Q"""
        visited = set()
        for i, sa in enumerate(state_action):
            if len(state_action) == self.max_episode_length:
                terminal_value = self.max_episode_length
            else:
                terminal_value = 0
            # only add returns for first time state-action pair appears
            if sa not in visited:
                visited.add(sa)
                self.returns[sa].append(len(state_action) - i + terminal_value)
                self.Q[sa] = np.mean(self.returns[sa])
        return self.returns

    def update_policy(self, state_action):
        for (previous_action, state, action) in set(state_action):
            self.policy[state] = max(self.actions(previous_action, state),
                                     key=(lambda a:self.Q[(previous_action, state, a)]))
        return self.policy

    def run(self, num_runs):
        run_durations = []
        for i in range(num_runs):
            state_action = self.episode()
            self.update_returns(state_action)
            self.update_policy(state_action)
            run_durations.append(len(state_action))
        return run_durations
                

def main():
    pend = ipg.InvertedPendulum(WINDOWDIMS, CARTDIMS, PENDULUMDIMS,
                                GRAVITY, A_CART)
    agent = MCESagent(pend, 2000)
    run_durations = agent.run(300)
    plt.scatter(range(len(run_durations)), run_durations)
    plt.show()

if __name__ == "__main__":
    main()
