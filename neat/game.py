import pickle
import pygame
from utils import get_color_array
from data import lines, border_lines
import neat
import numpy as np

from pygame.locals import (
    K_LEFT,
    K_RIGHT,
    K_UP,
    K_DOWN,
)

from sprites import Car

pygame.init()

screen = pygame.display.set_mode((1024, 768), flags=pygame.SCALED, vsync=1)
background = pygame.surfarray.make_surface(get_color_array(border_lines))

initial_point_x = lines[0][2]
initial_point_y = lines[0][3]
car = Car(1)

done = False
clock = pygame.time.Clock()
episode = 0
total_rewards = 0

with open("neat_genome", "rb") as f:
    winner = pickle.load(f)
config_file = 'config-feedforward'
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
net = neat.nn.FeedForwardNetwork.create(winner, config)

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # pressed = pygame.key.get_pressed()
    # car.update(pressed[K_LEFT], pressed[K_RIGHT], pressed[K_UP], pressed[K_DOWN])
    action = net.activate(car.get_state())
    car.update(*action)
    screen.blit(background, (0, 0))
    screen.blit(car.surf, car.rect)
    pygame.display.flip()
    clock.tick(60)
