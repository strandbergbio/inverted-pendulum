# Monte Carlo with Exploring Starts

import invertedPendulumGame as ipg
import numpy as np
#import matplotlib.pyplot as plt
import pygame
import sys, os, argparse
import pickle
from pygame.locals import *
from collections import defaultdict

WINDOWDIMS = (1200, 600)
CARTDIMS = (50, 10)
PENDULUMDIMS = (6, 200)
GRAVITY = 0.13
REFRESHFREQ = 100
A_CART = 0.15

class MCESagent(object):       
    def __init__(self, pendulum, max_episode_length, soft_policy_prob, alpha):
        """accepts bin divisions to create discretized states
        x_bins and theta_bins given in range 0 to 1
        v_bins and omega_bins given in actual values
        """
        assert isinstance(pendulum, ipg.InvertedPendulum)
        self.pendulum = pendulum
        self.max_episode_length = max_episode_length
        self.soft_policy_prob = soft_policy_prob
        self.alpha = alpha
        self.epsilon = 10 ** (-10)
        self.x_bins = [self.pendulum.CARTWIDTH / 2,
                       self.pendulum.WINDOWWIDTH - self.pendulum.CARTWIDTH / 2 - self.epsilon,
                       np.inf]
        self.coarse_xs = [-1, 0, 1]
        self.theta_bins = [-np.pi / 2, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2,
                           np.pi / 2 - self.epsilon, np.inf]
        self.coarse_thetas = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
        self.v_bins = [-10 * self.pendulum.A_CART,
                       -1 * self.pendulum.A_CART,
                       self.pendulum.A_CART - self.epsilon,
                       10 * self.pendulum.A_CART, np.inf]
        self.coarse_vs = [-2, -1, 0, 1, 2]
        self.omega_bins = [-0.01, -0.001, 0.001, 0.01, np.inf]
        self.coarse_omegas = [-2, -1, 0, 1, 2]
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
        """defines available actions given state and previous action-
        must stop moving in one direction before moving in the other
        """
        x, _, _, _ = coarse_state
        result = ['None']
        if previous_action != 'Right' and x != self.coarse_xs[0]:
            result.append('Left')
        if previous_action != 'Left' and x != self.coarse_xs[-1]:
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
        x = np.random.choice([self.pendulum.CARTWIDTH / 2,
                              self.pendulum.WINDOWWIDTH / 2,
                              self.pendulum.WINDOWWIDTH - self.pendulum.CARTWIDTH / 2])
        x = self.pendulum.WINDOWWIDTH / 2
        v = 0
        theta = np.random.uniform(-0.01, 0.01)
        omega = 0
        return is_dead, x, v, theta, omega

    def optimal_policy(self, previous_action, state):
        return max(self.actions(previous_action, state),
                   key=lambda a:self.Q[(previous_action,state,a)])

    def epsilon_policy(self, previous_action, state):
        if np.random.uniform(0,1) > self.soft_policy_prob:
            return self.optimal_policy(previous_action, state)
        else:
            return np.random.choice(self.actions(previous_action, state))

    def episode(self, init, policy = None):
        if policy is None:
            policy = self.epsilon_policy
        is_dead, x, v, theta, omega = init
        self.pendulum.set_state((is_dead, 0, x, v, theta, omega))
        previous_action = "None"
        is_dead, state = self.coarse_state()
        action = policy(previous_action, state)
        state_actions = []
        i = 0
        while not is_dead and i < self.max_episode_length:
            # Check the state you will go into
            self.pendulum.update_state(action)
            is_dead, new_state = self.coarse_state()
            old_Q = self.Q[(previous_action, state, action)]
            new_Q = self.Q[(action, new_state, self.optimal_policy(action, new_state))]
            # Update the value estimate
            self.Q[(previous_action, state, action)] += self.alpha * (1 + new_Q - old_Q)
            state_actions.append((previous_action, state, action))
            # Change your state
            previous_action, state = action, new_state
            action = policy(previous_action, state)
            i += 1
        return state_actions

    @staticmethod
    def plot_episode(state_action):
        theta_coarse = [theta for (_, (_, _, theta, _), _) in state_action]
        plt.scatter(range(len(theta_coarse)), theta_coarse)
        plt.show()

    def run(self, num_runs, train=False, policy=None):
        run_durations = []
        for i in range(num_runs):
            init = self.random_init() if train else self.nice_init()
            state_action = self.episode(init, policy=policy)
            run_durations.append(len(state_action))
        return run_durations

    def test(self):
        #assert self.fine_to_coarse(0, self.theta_bins, self.coarse_thetas) == 0
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
            action = self.agent.optimal_policy(previous_action, state)
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
    parser = argparse.ArgumentParser(
        description="Pass name of pickle to use or write to")
    parser.add_argument('filename', type=str)
    args = parser.parse_args()
    FILEPATH = os.getcwd() + "/" + args.filename + ".pickle"
    if os.path.exists(FILEPATH):
        with open(FILEPATH, 'rb') as f:
            agent = pickle.load(f)
    else:
        pend = ipg.InvertedPendulum(WINDOWDIMS, CARTDIMS, PENDULUMDIMS,
                                    GRAVITY, A_CART)        
        agent = MCESagent(pend, 20000, 0.1, 0.05)
        agent.test()
        performance = []
        for i in range(10):
            agent.run(100)
            durations = agent.run(50, train=False, policy=agent.optimal_policy)
            avg_duration = np.mean(durations)
            performance.append(avg_duration)
            print("Step: {}, Average Duration {}".format(i+1, avg_duration))
    ##        if i % 50 == 0:
    ##            e = agent.episode(agent.nice_init())
    ##            agent.plot_episode(e)
        with open(FILEPATH, 'wb') as output:
            pickle.dump(agent, output)
    game = MCESgame(WINDOWDIMS, CARTDIMS, PENDULUMDIMS, GRAVITY, A_CART,
                    REFRESHFREQ, agent)
    game.game()

if __name__ == "__main__":
    main() 
