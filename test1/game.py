import pygame

from pygame.locals import (
    K_LEFT,
    K_RIGHT,
    K_UP,
    K_DOWN,
)

from sprites import Car

pygame.init()

screen = pygame.display.set_mode((1024, 768), flags=pygame.SCALED, vsync=1)

car = Car(512, 384, 1, 0.5, 7)

done = False
clock=pygame.time.Clock()

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    pressed = pygame.key.get_pressed()
    car.update(pressed[K_LEFT], pressed[K_RIGHT], pressed[K_UP], pressed[K_DOWN])
    screen.fill((255, 255, 255))
    screen.blit(car.surf, car.rect)
    pygame.display.flip()
    clock.tick(60)
