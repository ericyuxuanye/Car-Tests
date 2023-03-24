import pygame
from math import sin, cos, pi, atan2, sqrt
from utils import (
    calc_distance_from_start,
    car_touching_line,
    get_distances,
    relative_car_velocities,
    num_lines,
)


class Car(pygame.sprite.Sprite):
    def __init__(self, init_x, init_y, acceleration, friction, rot_speed, live=True):
        if live:
            super(Car, self).__init__()
            self.normal = pygame.image.load("car.png").convert_alpha()
        self.live = live
        self.width = 24
        self.height = 47
        self.init_x = init_x
        self.init_y = init_y
        self.rot_speed = rot_speed
        self.accel = acceleration
        self.friction = friction
        self.just_hit = False
        self.reset()

    def reset(self):
        self.x = self.init_x
        self.y = self.init_y
        self.velocity = [0.0, 0.0]
        self.prev_distance = 0
        if self.live:
            self.surf = self.normal
            self.rect = self.surf.get_rect(center=(self.x, self.y))
        self.rotation = 0

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
        if car_touching_line(self.x, self.y, self.width, self.height, self.rotation):
            self.just_hit = True
            self.reset()
            return
        if self.live:
            self.surf = pygame.transform.rotate(self.normal, self.rotation)
            self.rect = self.surf.get_rect(center=(round(self.x), round(self.y)))

    def get_state(self):
        state = []
        distances = get_distances((self.x, self.y), self.rotation)
        state.extend(distances)
        velocities = relative_car_velocities(self.velocity, self.rotation)
        state.extend(velocities)
        return state

    def get_reward(self):
        curr_distance = calc_distance_from_start(
            (self.x, self.y)
        )
        reward = curr_distance - self.prev_distance
        if reward > num_lines / 2:
            reward = curr_distance - num_lines - self.prev_distance
        elif reward < -num_lines / 2:
            reward = (num_lines - self.prev_distance) + curr_distance
        self.prev_distance = curr_distance
        reward *= 50
        if reward <= 0:
            # So that the agent does not stay in place
            reward = -0.1
        # So that agent does not stay in place
        # reward -= 0.1
        if self.just_hit:
             self.just_hit = False
        return reward
