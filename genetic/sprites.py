import pygame
import numpy as np
import random
from math import sin, cos, pi, atan2, sqrt
from utils import (
    calc_distance_from_start,
    car_touching_line,
    get_distances,
    relative_car_velocities,
    line_angle,
)
from data import num_lines, lines, border_lines


class Car(pygame.sprite.Sprite):
    def __init__(self, acceleration, friction, rot_speed):
        super(Car, self).__init__()
        self.normal = pygame.image.load("car.png").convert_alpha()
        self.width = 24
        self.height = 47
        self.rot_speed = rot_speed
        self.accel = acceleration
        self.friction = friction
        self.just_hit = False
        self.velocity = np.empty(2)
        self.reset()

    def reset(self):
        idx = random.randint(0, len(lines)-1)
        line = lines[idx]
        self.x = line[2]
        self.y = line[3]
        self.velocity[:] = [0.0, 0.0]
        self.prev_distance = idx
        self.rotation = line_angle(line)
        self.surf = pygame.transform.rotate(self.normal, self.rotation)
        self.rect = self.surf.get_rect(center=(self.x, self.y))

    def update(self, left, right, forward, backward):
        if left:
            self.rotation += self.rot_speed
        if right:
            self.rotation -= self.rot_speed

        radians = self.rotation / 180 * pi + pi / 2
        if forward:
            self.velocity[0] += self.accel * cos(radians)
            # subtract because y is flipped
            self.velocity[1] -= self.accel * sin(radians)
        if backward:
            self.velocity[0] -= self.accel * cos(radians)
            self.velocity[1] += self.accel * sin(radians)
        # friction calculation
        r = sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
        theta = atan2(self.velocity[1], self.velocity[0])
        r = max(r - self.friction, 0)
        self.velocity[0] = r * cos(theta)
        self.velocity[1] = r * sin(theta)

        self.x += self.velocity[0]
        self.y += self.velocity[1]
        if car_touching_line(self.x, self.y, self.width, self.height, self.rotation, lines, border_lines):
            self.just_hit = True
            self.reset()
            return
        self.surf = pygame.transform.rotate(self.normal, self.rotation)
        self.rect = self.surf.get_rect(center=(round(self.x), round(self.y)))

    def get_state(self):
        state = np.empty((10,), dtype=np.float32)
        distances = get_distances((self.x, self.y), self.rotation, border_lines)
        state[:8] = distances
        velocities = relative_car_velocities(self.velocity, self.rotation)
        state[8:10] = velocities
        return state

    def get_reward(self):
        curr_distance = calc_distance_from_start(
            (self.x, self.y),
            lines
        )
        reward = curr_distance - self.prev_distance
        if reward > num_lines / 2:
            reward = curr_distance - num_lines - self.prev_distance
        elif reward < -num_lines / 2:
            reward = (num_lines - self.prev_distance) + curr_distance
        self.prev_distance = curr_distance
        reward *= 50
        # if reward <= 0:
        #     # So that the agent does not stay in place
        #     reward = -0.1
        # So that agent does not stay in place
        # reward -= 0.1
        if self.just_hit:
             self.just_hit = False
        return reward
