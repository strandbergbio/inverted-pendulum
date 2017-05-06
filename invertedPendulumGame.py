import pygame, sys
import numpy as np
from pygame.locals import *

pygame.init()
mainClock = pygame.time.Clock()

WINDOWWIDTH = 1000
WINDOWHEIGHT = 600
windowSurface = pygame.display.set_mode((WINDOWWIDTH,WINDOWHEIGHT),0,32)
pygame.display.set_caption('Inverted Pendulum Game')

basicFont = pygame.font.SysFont(None, 48)
BLACK = (0,0,0)
WHITE = (255,255,255)

CARTWIDTH = 50
CARTHEIGHT = 10
PENDULUMLENGTH = 200
PENDULUMWIDTH = 6
GRAVITY = 0.15
REFRESHFREQ = 50

time = 0
x_cart = windowSurface.get_rect().centerx
Y_CART = (3 * WINDOWHEIGHT / 4)
V_CART = 5.0
theta_pend = 0.01
omega_pend = 0.0
move = "None"

def update_state(t, x, theta, omega, action):
    dead = np.pi / 2
    if abs(theta) >= dead:
        return t, x, theta, 0
    t += 1
    if action == "Left" and x > (CARTWIDTH / 2):
        x -= V_CART
        # cart motion adds extra angular velocity
        theta -= V_CART * np.cos(theta) / float(PENDULUMLENGTH)
    elif action == "Right" and x < (WINDOWWIDTH - (CARTWIDTH / 2)):
        x += V_CART
        theta += V_CART * np.cos(theta) / float(PENDULUMLENGTH)
    theta += omega
    # torque from gravity
    omega += GRAVITY * np.sin(theta) / float(PENDULUMLENGTH)
    return t, x, theta, omega

def rotation_matrix(theta):
    return np.array([[np.cos(theta), np.sin(theta)],
                     [-1 * np.sin(theta), np.cos(theta)]])

def drawCart(x, theta, surface):
    cart = pygame.Rect(x - CARTWIDTH // 2, Y_CART, CARTWIDTH, CARTHEIGHT)
    pygame.draw.rect(surface, BLACK, cart)
    static_pendulum_array = np.array(
        [[-PENDULUMWIDTH / 2, 0],
         [PENDULUMWIDTH / 2, 0],
         [PENDULUMWIDTH / 2, -PENDULUMLENGTH],
         [-PENDULUMWIDTH / 2, -PENDULUMLENGTH]]).T
    pendulum_array = np.dot(rotation_matrix(theta), static_pendulum_array)
    pendulum_array += np.array([[x],[Y_CART]])
    pendulum = pygame.draw.polygon(surface, BLACK,
        ((pendulum_array[0,0],pendulum_array[1,0]),
         (pendulum_array[0,1],pendulum_array[1,1]),
         (pendulum_array[0,2],pendulum_array[1,2]),
         (pendulum_array[0,3],pendulum_array[1,3])))


while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN:
            if event.key == K_LEFT:
                move = "Left"
            if event.key == K_RIGHT:
                move = "Right"
        if event.type == KEYUP:
            if event.key == K_LEFT:
                move = "None"
            if event.key == K_RIGHT:
                move = "None"
            if event.key == K_ESCAPE:
                pygame.quit()
                sys.exit()
    time, x_cart, theta_pend, omega_pend = update_state(time, x_cart,
                                                        theta_pend,
                                                        omega_pend, move)
    time_text = basicFont.render('t = {} s'.format(time / float(REFRESHFREQ)),
                                 True, BLACK, WHITE)
    time_text_rect = time_text.get_rect()
    time_text_rect.center = (0.5 * WINDOWWIDTH, 0.5 * WINDOWHEIGHT)
    windowSurface.blit(time_text, time_text_rect)
    
    windowSurface.fill(WHITE)
    drawCart(x_cart, theta_pend, windowSurface)
    pygame.display.update()

    mainClock.tick(REFRESHFREQ)
