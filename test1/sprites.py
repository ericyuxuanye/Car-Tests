import pygame
from math import sin, cos, pi, atan2, sqrt


class Car(pygame.sprite.Sprite):
    def __init__(self, x, y, acceleration, friction, rot_speed):
        super(Car, self).__init__()
        self.normal = pygame.image.load("car.png").convert_alpha()
        self.velocity = [0., 0.]
        self.surf = self.normal
        self.x = x
        self.y = y
        self.rect = self.surf.get_rect(center=(x, y))
        self.rotation = 0
        self.rot_speed = rot_speed
        self.accel = acceleration
        self.friction = friction

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
        self.surf = pygame.transform.rotate(self.normal, self.rotation)
        self.rect = self.surf.get_rect(center=(round(self.x), round(self.y)))
