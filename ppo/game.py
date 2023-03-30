import pygame
from utils import get_color_array, border_lines, lines
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

pygame.init()

screen = pygame.display.set_mode((1024, 768), flags=pygame.SCALED, vsync=1)
background = pygame.surfarray.make_surface(get_color_array(border_lines))

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