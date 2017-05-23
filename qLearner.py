# Monte Carlo with Exploring Starts

import invertedPendulumGame as ipg
import numpy as np
import matplotlib.pyplot as plt
import pygame
import sys
from pygame.locals import *
from collection import defaultdict

WINDOWDIMS = (1200, 600)
CARTDIMS = (50, 10)
PENDULUMDIMS = (6, 200)
GRAVITY = 0.13
REFRESHFREQ = 100
A_CART = 0.15

class MCESagent(object):       
    def __init__(self, pendulum, max_episode_length, soft_policy_prob):
        """accepts bin divisions to create discretized states
        x_bins and theta_bins given in range 0 to 1
        v_bins and omega_bins given in actual values
        """
        assert isinstance(pendulum, ipg.InvertedPendulum)
        self.pendulum = pendulum
        self.max_episode_length = max_episode_length
        self.soft_policy_prob = soft_policy_prob
        self.epsilon = 10 ** (-10)
        self.x_bins = [self.pendulum.CARTWIDTH / 2,
                       self.pendulum.WINDOWWIDTH / 4,
                       self.pendulum.WINDOWWIDTH * 3 / 4,
                       self.pendulum.WINDOWWIDTH - self.pendulum.CARTWIDTH / 2 - self.epsilon,
                       np.inf]
        self.coarse_xs = [-2, -1, 0, 1, 2]
        self.theta_bins = [-np.pi / 2, -0.3, -0.05, 0.05, 0.3,
                           np.pi / 2 - self.epsilon, np.inf]
        self.coarse_thetas = [-3, -2, -1, 0, 1, 2, 3]
        self.v_bins = [-10 * self.pendulum.A_CART,
                       -1 * self.pendulum.A_CART,
                       self.pendulum.A_CART - self.epsilon,
                       10 * self.pendulum.A_CART, np.inf]
        self.coarse_vs = [-2, -1, 0, 1, 2]
        self.omega_bins = [-0.01, -0.001, 0.001, 0.01, np.inf]
        self.coarse_omegas = [-2, -1, 0, 1, 2]
        self.policy = dict()
        self.Q = dict()

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
        result = ['None']
        if previous_action != 'Right' and x != self.coarse_xs[-1]:
            result.append('Left')
        if previous_action != 'Left' and x != self.coarse_xs[0]:
            result.append('Right')
        return result

    def random_init(self):
        x = np.random.uniform(self.pendulum.CARTWIDTH / 2,
                                    self.pendulum.WINDOWWIDTH - self.pendulum.CARTWIDTH / 2)
        theta = np.random.uniform(-np.pi / 2, np.pi / 2)
        v = np.random.normal(0, 1.0)
        omega = np.random.normal(0, 0.1)
        is_dead = False
        return is_dead, x, v, theta, omega

    def nice_init(self):
        is_dead = False
        x = self.pendulum.WINDOWWIDTH / 2
        v = 0
        theta = np.random.uniform(-0.01, 0.01)
        omega = 0
        return is_dead, x, v, theta, omega

    def episode(self, init):
        is_dead, x, v, theta, omega = init
        self.pendulum.set_state((is_dead, 0, x, v, theta, omega))
        is_dead, state = self.coarse_state()
        state_action = []
        previous_action = np.random.choice(self.actions("None", state))
        i = 0
        while not is_dead and i < self.max_episode_length:
            i += 1
            p = np.random.uniform(0,1)
            if (previous_action, state) in self.policy and p > self.soft_policy_prob:
                action = self.policy[(previous_action, state)]
            else:
                action = np.random.choice(self.actions(previous_action, state))
            self.pendulum.update_state(action)
            is_dead, state = self.coarse_state()
            state_action.append((previous_action, state, action))
            previous_action = action
        return state_action

    @staticmethod
    def plot_episode(state_action):
        theta_coarse = [theta for (_, (_, _, theta, _), _) in state_action]
        plt.scatter(range(len(theta_coarse)), theta_coarse)
        plt.show()

    def update_returns(self, state_action, visit="every-visit"):
        """ updates self.returns AND self.Q"""
        if visit == "first-visit":
            visited = set()
        if len(state_action) == self.max_episode_length:
            terminal_value = self.max_episode_length
        else:
            terminal_value = 0
        for i, sa in enumerate(state_action):
            if sa not in self.Q:
                self.Q[sa] = np.array([0,0])
            if visit == "first-visit":
                if sa not in visited:
                    visited.add(sa)
                    self.Q[sa] += [len(state_action) - i + terminal_value, 1]
            elif visit == "every-visit":
                self.Q[sa] += [len(state_action) - i + terminal_value, 1]
            else:
                raise ValueError("Invalid argument for visit in update_returns")
        return self.Q

    def update_policy(self, state_action):       
        for (previous_action, state, action) in set(state_action):
            actions = self.actions(previous_action, state)
            def optimal(action):
                if (previous_action, state, action) in self.Q:
                    q = self.Q[(previous_action, state, action)]
                else:
                    q = np.array([0,1])
                return q[0] / float(q[1])
            self.policy[(previous_action, state)] = max(self.actions(previous_action, state),
                                                        key=optimal)
        return self.policy

    def run(self, num_runs, train=True):
        run_durations = []
        for i in range(num_runs):
            init = self.random_init() if train else self.nice_init()
            state_action = self.episode(init)
            self.update_returns(state_action)
            self.update_policy(state_action)
            run_durations.append(len(state_action))
        return run_durations

    def test(self):
        assert self.fine_to_coarse(0, self.theta_bins, self.coarse_thetas) == 0
        assert self.fine_to_coarse(-3 * A_CART, self.v_bins, self.coarse_vs) == -1
        assert self.actions("Left", (0, 0, 0, 0)) == ["None", "Left"]
        
        print("Tests passed")


class MCESgame(ipg.InvertedPendulumGame):
    def __init__(self, windowdims, cartdims, penddims,
                 gravity, a_cart, refreshfreq, agent):
        assert isinstance(agent, MCESagent)
        self.agent = agent
        super(MCESgame, self).__init__(windowdims, cartdims, penddims,
                                       gravity, a_cart, refreshfreq,
                                       agent.pendulum)

    def game_round(self):
        self.agent.pendulum.reset_state()
        previous_action = "None"
        while not self.agent.pendulum.is_dead:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYUP:
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        sys.exit()
            is_dead, state = self.agent.coarse_state()
            if (previous_action, state) in self.agent.policy:
                action = self.agent.policy[(previous_action, state)]
            else:
                action = np.random.choice(self.agent.actions(previous_action, state))
            t, x, _, theta, _ = self.agent.pendulum.update_state(action)
            previous_action = action
            self.time = t    
            self.surface.fill(self.WHITE)
            self.draw_cart(x, theta)

            time_text = "t = {}".format(self.time_seconds())
            self.render_text(time_text, (0.1 * self.WINDOWWIDTH, 0.1 * self.WINDOWHEIGHT),
                             position = "topleft", fontsize = 40)
            if action == 'Left':
                action_text = '<-       '
            elif action == 'None':
                action_text = '    |    '
            elif action == 'Right':
                action_text = '       ->'
            self.render_text(action_text, (0.1 * self.WINDOWWIDTH, 0.2 * self.WINDOWHEIGHT),
                             position = "topleft", fontsize = 40)
            state_text = str(state)
            self.render_text(state_text, (0.1 * self.WINDOWWIDTH, 0.3 * self.WINDOWHEIGHT),
                             position = "topleft", fontsize = 40)
            
            pygame.display.update()
            self.clock.tick(self.REFRESHFREQ)
        if (self.time_seconds()) > self.high_score:
            self.high_score = self.time_seconds()

    def game(self):
        self.starting_page()
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        sys.exit()
            self.game_round()
            self.end_of_round()
            pygame.time.wait(500)
        
def main():
    pend = ipg.InvertedPendulum(WINDOWDIMS, CARTDIMS, PENDULUMDIMS,
                                GRAVITY, A_CART)        
    agent = MCESagent(pend, 2000, 0.05)
    agent.test()
    performance = []
    for i in range(1):
        agent.run(200)
        durations = agent.run(200, train=False)
        avg_duration = np.mean(durations)
        performance.append(avg_duration)
        print "Step: {}, Average Duration {}".format(i+1, avg_duration)
##        if i % 50 == 0:
##            e = agent.episode(agent.nice_init())
##            agent.plot_episode(e)
##        if i % 150 == 0:
##            print agent.policy
    game = MCESgame(WINDOWDIMS, CARTDIMS, PENDULUMDIMS, GRAVITY, A_CART,
                    REFRESHFREQ, agent)
    game.game()

if __name__ == "__main__":
    main() 
