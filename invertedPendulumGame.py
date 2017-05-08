import pygame, sys
import numpy as np
from pygame.locals import *

WINDOWDIMS = (1000, 700)
CARTDIMS = (50, 10)
PENDULUMDIMS = (6, 200)
GRAVITY = 0.13
REFRESHFREQ = 100
A_CART = 0.16

class InvertedPendulum(object):
    def __init__(self, windowdims, cartdims, penddims, gravity, a_cart):
        self.WINDOWWIDTH = windowdims[0]
        self.WINDOWHEIGHT = windowdims[1]

        self.CARTWIDTH = cartdims[0]
        self.CARTHEIGHT = cartdims[1]
        self.PENDULUMWIDTH = penddims[0]
        self.PENDULUMLENGTH = penddims[1]

        self.GRAVITY = gravity

        self.is_dead = False
        self.time = 0
        self.x_cart = self.WINDOWWIDTH / 2
        self.Y_CART = 3 * self.WINDOWHEIGHT / 4
        self.v_cart = 0
        self.A_CART = a_cart
        self.theta = np.random.uniform(-0.01,0.01)
        self.omega = np.random.uniform(-0.01,0.01)
        self.move = "None"

    def update_state(self, action):
        assert isinstance(action, str)
        if self.is_dead:
            raise RuntimeError("tried to call update_state while state was dead")
        self.time += 1
        self.x_cart += self.v_cart
        if self.x_cart < 0:
            self.x_cart = 0
            self.v_cart = 0
        elif self.x_cart > self.WINDOWWIDTH:
            self.x_cart = self.WINDOWWIDTH
            self.v_cart = 0
        self.theta += self.omega + self.v_cart * np.cos(self.theta) / float(self.PENDULUMLENGTH)
        self.omega += self.GRAVITY * np.sin(self.theta) / float(self.PENDULUMLENGTH)
        if action == "Left":
            self.v_cart -= self.A_CART
        elif action == "Right":
            self.v_cart += self.A_CART
        elif action == "None":
            self.v_cart = 0
        else:
            raise RuntimeError("action must be 'Left', 'Right', or 'None'")
        if abs(self.theta) >= np.pi / 2:
            self.is_dead = True

class InvertedPendulumGame(InvertedPendulum):
    def __init__(self, windowdims, cartdims, penddims, gravity, a_cart, refreshfreq):
        InvertedPendulum.__init__(self, windowdims, cartdims, penddims, gravity, a_cart)
        pygame.init()
        self.clock = pygame.time.Clock()
        self.REFRESHFREQ = refreshfreq
        self.surface = pygame.display.set_mode(windowdims,0,32)
        pygame.display.set_caption('Inverted Pendulum Game')
        self.font = pygame.font.SysFont(None, 48)
        self.BLACK = (0,0,0)
        self.WHITE = (255,255,255)
        self.static_pendulum_array = np.array(
            [[-self.PENDULUMWIDTH / 2, 0],
             [self.PENDULUMWIDTH / 2, 0],
             [self.PENDULUMWIDTH / 2, -self.PENDULUMLENGTH],
             [-self.PENDULUMWIDTH / 2, -self.PENDULUMLENGTH]]).T

    def draw_cart(self):
        cart = pygame.Rect(self.x_cart - self.CARTWIDTH // 2, self.Y_CART, self.CARTWIDTH, self.CARTHEIGHT)
        pygame.draw.rect(self.surface, self.BLACK, cart)
        pendulum_array = np.dot(self.rotation_matrix(self.theta), self.static_pendulum_array)
        pendulum_array += np.array([[self.x_cart],[self.Y_CART]])
        pendulum = pygame.draw.polygon(self.surface, self.BLACK,
            ((pendulum_array[0,0],pendulum_array[1,0]),
             (pendulum_array[0,1],pendulum_array[1,1]),
             (pendulum_array[0,2],pendulum_array[1,2]),
             (pendulum_array[0,3],pendulum_array[1,3])))

    @staticmethod
    def rotation_matrix(theta):
        return np.array([[np.cos(theta), np.sin(theta)],
                         [-1 * np.sin(theta), np.cos(theta)]])

    def starting_page(self):
        self.surface.fill(WHITE)
        title = self.font.render("Inverted Pendulum Game", True, BLACK, WHITE)
        title_rect = title.get_rect()
        title_rect.center = self.surface.center
        self.surface.blit(title, title_rect)

    def run_game(self):
        action = "None"
        while not self.is_dead:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == K_LEFT:
                        action = "Left"
                    if event.key == K_RIGHT:
                        action = "Right"
                if event.type == KEYUP:
                    if event.key == K_LEFT:
                        action = "None"
                    if event.key == K_RIGHT:
                        action = "None"
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        sys.exit()
            self.update_state(action)
    
            self.surface.fill(self.WHITE)
            self.draw_cart()
    
            time_text = self.font.render('t = {}'.format(self.time / float(self.REFRESHFREQ)),
                                         True, self.BLACK, self.WHITE)
            time_text_rect = time_text.get_rect()
            time_text_rect.topleft = (0.1 * self.WINDOWWIDTH, 0.1 * self.WINDOWHEIGHT)
            self.surface.blit(time_text, time_text_rect)
            
            pygame.display.update()
            self.clock.tick(self.REFRESHFREQ)
  

def main():
    inv = InvertedPendulumGame(WINDOWDIMS, CARTDIMS, PENDULUMDIMS, GRAVITY, A_CART, REFRESHFREQ)
    inv.run_game()
main()
