import pygame
from math import sin, cos, pi, atan2, sqrt
from utils import car_touching_line, get_straight_coordinates, relative_car_velocities


class Car(pygame.sprite.Sprite):
    def __init__(self, init_x, init_y, acceleration, friction, rot_speed):
        super(Car, self).__init__()
        self.normal = pygame.image.load("car.png").convert_alpha()
        self.init_x = init_x
        self.init_y = init_y
        self.rot_speed = rot_speed
        self.accel = acceleration
        self.friction = friction
        self.reset()

    def reset(self):
        self.x = self.init_x
        self.y = self.init_y
        self.velocity = [0., 0.]
        self.surf = self.normal
        self.rect = self.surf.get_rect(center=(self.x, self.y))
        self.width = self.rect.width
        self.height = self.rect.height
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
            self.reset()
            return
        self.surf = pygame.transform.rotate(self.normal, self.rotation)
        self.rect = self.surf.get_rect(center=(round(self.x), round(self.y)))

    def laser_coordinates(self):
        return get_straight_coordinates((self.x, self.y), self.rotation)
