import pygame
from utils import fill_in_line, border_lines, lines
from model import get_action, update_model, save_model
import signal
import traceback

from pygame.locals import (
    K_LEFT,
    K_RIGHT,
    K_UP,
    K_DOWN,
)

LIVE = False

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

def handler(signum, frame):
    traceback.print_stack(frame)
    save_model()
    exit()

signal.signal(signal.SIGINT, handler)

if LIVE:
    pygame.init()

    screen = pygame.display.set_mode((1024, 768))
    background = fill_in_line(border_lines)

initial_point_x = lines[0][2]
initial_point_y = lines[0][3]
car = Car(initial_point_x, initial_point_y, 1.5, 1, 7, LIVE)

done = False
clock = pygame.time.Clock()
episode = 0
total_rewards = 0

while not done:
    if LIVE:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    # pressed = pygame.key.get_pressed()
    # car.update(pressed[K_LEFT], pressed[K_RIGHT], pressed[K_UP], pressed[K_DOWN])
    action = action_to_keys[get_action(car.get_state())]
    car.update(*action)
    just_hit = car.just_hit
    if just_hit:
        print("episode:", episode, end="\t")
        print("rewards:", total_rewards)
        episode += 1
        total_rewards = 0

    if LIVE:
        screen.blit(background, (0, 0))
        screen.blit(car.surf, car.rect)
        pygame.display.flip()
        clock.tick(60)

    # update model
    reward = car.get_reward()
    total_rewards += reward
    if not LIVE:
        update_model(car.get_state(), reward, just_hit)

save_model()
