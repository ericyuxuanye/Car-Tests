import pygame
import numpy
import pygame.pixelcopy
from utils import get_color_array
from data import lines, border_lines
from stable_baselines3 import PPO

from pygame.locals import (
    K_LEFT,
    K_RIGHT,
    K_UP,
    K_DOWN,
)

from sprites import Car

action_to_keys = [
    (0, 0, 0, 0),
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
    (1, 0, 0, 1),
    (1, 0, 1, 0),
    (0, 1, 1, 0),
    (0, 1, 0, 1),
]

def make_surface_rgba(array):
    """Returns a surface made from a [w, h, 4] numpy array with per-pixel alpha
    """
    shape = array.shape
    if len(shape) != 3 and shape[2] != 4:
        raise ValueError("Array not RGBA")

    # Create a surface the same width and height as array and with
    # per-pixel alpha.
    surface = pygame.Surface(shape[0:2], pygame.SRCALPHA, 32)

    # Copy the rgb part of array to the new surface.
    pygame.pixelcopy.array_to_surface(surface, array[:,:,0:3])

    # Copy the alpha part of array to the surface using a pixels-alpha
    # view of the surface.
    surface_alpha = numpy.array(surface.get_view('A'), copy=False)
    surface_alpha[:,:] = array[:,:,3]

    return surface

pygame.init()

screen = pygame.display.set_mode((1024, 768), flags=pygame.SCALED, vsync=1)
bg_image = pygame.image.load("background.jpg")
background = pygame.Surface((1024, 768), pygame.SRCALPHA)
background.blit(bg_image, (0, 0))
track = make_surface_rgba(get_color_array(border_lines))
background.blit(track, (0, 0))

initial_point_x = lines[0][2]
initial_point_y = lines[0][3]
car = Car(1.5, 1, 7)

done = False
clock = pygame.time.Clock()
episode = 0
total_rewards = 0

model = PPO.load("ppo_racer")

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # pressed = pygame.key.get_pressed()
    # car.update(pressed[K_LEFT], pressed[K_RIGHT], pressed[K_UP], pressed[K_DOWN])
    action = action_to_keys[model.predict(car.get_state())[0]]
    car.update(*action)
    screen.blit(background, (0, 0))
    screen.blit(car.surf, car.rect)
    pygame.display.flip()
    clock.tick(60)
