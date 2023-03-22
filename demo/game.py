import pygame
from utils import fill_in_line, border_lines, lines

from pygame.locals import (
    K_LEFT,
    K_RIGHT,
    K_UP,
    K_DOWN,
)

from sprites import Car

pygame.init()

screen = pygame.display.set_mode((1024, 768))
background = fill_in_line(border_lines)

initial_point_x = lines[0][2]
initial_point_y = lines[0][3]
car = Car(initial_point_x, initial_point_y, 1.5, 1, 7)

done = False
clock = pygame.time.Clock()

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    pressed = pygame.key.get_pressed()
    car.update(pressed[K_LEFT], pressed[K_RIGHT], pressed[K_UP], pressed[K_DOWN])
    screen.blit(background, (0, 0))
    screen.blit(car.surf, car.rect)
    # laser_coordinates = car.laser_coordinates()
    # print(forward_coordinate)
    # for point in laser_coordinates:
    #     pygame.draw.circle(screen, "black", point, 5)
    pygame.display.flip()
    clock.tick(60)
