import pygame
from math import sqrt
import numpy as np
from numba import njit

from typing import List, Tuple

pygame.init()

screen = pygame.display.set_mode((1024, 768), flags=pygame.SCALED, vsync=1)
points: List[Tuple[int, int]] = []
# parametric line (x slope, y slope, init_x, init_y)
lines: List[Tuple[int, int, int, int]] = []
border_lines: List[Tuple[float, float, float, float]] = []
# measures distance from edge of track to center
width = 40

clock = pygame.time.Clock()
done = False
finished = False

fill_surf = pygame.surface.Surface((1024, 768))
should_fill = np.zeros((1024, 768), dtype=bool)
fill_surf.fill("white")


@njit
def line_intersect(line1, line2):
    a, c, b, d = line1
    f, h, g, j = line2
    t = -(d * f - b * h + g * h - f * j) / (c * f - a * h)
    return a * t + b, c * t + d


@njit
def line_dist_sq(point, line):
    vec_const = (line[2] - point[0], line[3] - point[1])
    t_comp = line[0] * line[0] + line[1] * line[1]
    c = vec_const[0] * line[0] + vec_const[1] * line[1]
    t = -c / t_comp
    if t > 1:
        t = 1
    elif t < 0:
        t = 0
    x = t * line[0] + line[2]
    y = t * line[1] + line[3]
    return (point[0] - x) ** 2 + (point[1] - y) ** 2


@njit
def line_dist_sq_np(x, y, line):
    vec_const = (line[2] - x, line[3] - y)
    t_comp = line[0] ** 2 + line[1] ** 2
    c = vec_const[0] * line[0] + vec_const[1] * line[1]
    t = -c / t_comp
    t[t > 1] = 1
    t[t < 0] = 0
    a = t * line[0] + line[2]
    b = t * line[1] + line[3]
    return (x - a) ** 2 + (y - b) ** 2


def fill(point1, point2):
    global should_fill
    global fill_surf
    change_x = point2[0] - point1[0]
    change_y = point2[1] - point1[1]
    lines.append((change_x, change_y, point1[0], point1[1]))
    x = np.arange(1024)
    y = np.arange(768)
    X, Y = np.meshgrid(x, y, indexing="ij")
    should_fill = should_fill | (line_dist_sq_np(X, Y, lines[-1]) <= width**2)
    color_array = np.full((1024, 768, 3), 255)
    color_array[should_fill, :] = 128
    fill_surf = pygame.surfarray.make_surface(color_array)


def did_intersect_np(line1, line2):
    """
    Line1 has t>=0, and is assumed to be an np array with size 4 on the last dimension
    Line2 has 0<=s<=1, and is a tuple
    """
    a = line1[..., 0]
    c = line1[..., 1]
    b = line1[..., 2]
    d = line1[..., 3]
    f, h, g, j = line2
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = c * f - a * h
        t = -(d * f - b * h + g * h - f * j) / denom
        s = (c * t + d - j) / h
    res = (denom != 0) & (t >= 0) & (0 <= s) & (s <= 1)
    return res

def fill_in_line():
    intersect_count = np.zeros((1024, 768))
    x = np.arange(1024)
    y = np.arange(768)
    X, Y = np.meshgrid(x, y, indexing="ij")
    point_line = np.empty((1024, 768, 4))
    point_line[X, Y, 2] = X
    point_line[X, Y, 3] = Y
    point_line[X, Y, 0] = 1
    point_line[X, Y, 1] = 0
    for line in border_lines:
        intersect_count[did_intersect_np(point_line, line)] += 1
    color_array = np.full((1024, 768, 3), 255)
    color_array[intersect_count % 2 == 1, :] = 128
    global fill_surf
    fill_surf = pygame.surfarray.make_surface(color_array)

def save():
    """
    Saves the file
    
    First line contains n: number of line segments
    next n lines: dx dy x y for each middle line
    next 2n lines: dx dy x y for each border line
    """
    with open("lines.txt", "wt") as f:
        print(len(lines), file=f)
        for line in lines:
            print(*line, file=f)
        for line in border_lines:
            print(*line, file=f)


while not done:
    pos = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.MOUSEBUTTONDOWN and not finished:
            points.append(pos)
            if len(points) > 1:
                change_x = points[-1][0] - points[-2][0]
                change_y = points[-1][1] - points[-2][1]
                lines.append((change_x, change_y, points[-2][0], points[-2][1]))
                # draw a line
                # fill(points[-2], points[-1])
        elif (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_SPACE
            and len(points) > 1
        ):
            # fill(points[-1], points[0])
            change_x = points[0][0] - points[-1][0]
            change_y = points[0][1] - points[-1][1]
            lines.append((change_x, change_y, points[-1][0], points[-1][1]))
            perp_lines: List[Tuple[float, float, int, int]] = []
            for line in lines:
                a = line[0]
                c = line[1]
                norm = sqrt(a**2 + c**2)
                multiplier = width / norm
                perp_lines.append((-c * multiplier, a * multiplier, line[2], line[3]))
            outer_points: List[Tuple[float, float]] = []
            inner_points: List[Tuple[float, float]] = []
            for i in range(len(points)):
                if i == 0:
                    prev = perp_lines[-1]
                    prev_line = lines[-1]
                else:
                    prev = perp_lines[i - 1]
                    prev_line = lines[i - 1]
                curr = perp_lines[i]
                curr_line = lines[i]
                outer_line_1_point = (prev[0] + prev[2], prev[1] + prev[3])
                outer_line_2_point = (curr[0] + curr[2], curr[1] + curr[3])
                inner_line_1_point = (-prev[0] + prev[2], -prev[1] + prev[3])
                inner_line_2_point = (-curr[0] + curr[2], -curr[1] + curr[3])
                intersection_outer = line_intersect(
                    (
                        prev_line[0],
                        prev_line[1],
                        outer_line_1_point[0],
                        outer_line_1_point[1],
                    ),
                    (
                        curr_line[0],
                        curr_line[1],
                        outer_line_2_point[0],
                        outer_line_2_point[1],
                    ),
                )
                outer_points.append(intersection_outer)
                intersection_inner = line_intersect(
                    (
                        prev_line[0],
                        prev_line[1],
                        inner_line_1_point[0],
                        inner_line_1_point[1],
                    ),
                    (
                        curr_line[0],
                        curr_line[1],
                        inner_line_2_point[0],
                        inner_line_2_point[1],
                    ),
                )
                inner_points.append(intersection_inner)
            for i in range(len(points) - 1):
                outer_line = (
                    outer_points[i + 1][0] - outer_points[i][0],
                    outer_points[i + 1][1] - outer_points[i][1],
                    outer_points[i][0],
                    outer_points[i][1],
                )
                inner_line = (
                    inner_points[i + 1][0] - inner_points[i][0],
                    inner_points[i + 1][1] - inner_points[i][1],
                    inner_points[i][0],
                    inner_points[i][1],
                )
                border_lines.append(outer_line)
                border_lines.append(inner_line)
            final_outer_line = (
                outer_points[0][0] - outer_points[-1][0],
                outer_points[0][1] - outer_points[-1][1],
                outer_points[-1][0],
                outer_points[-1][1],
            )
            final_inner_line = (
                inner_points[0][0] - inner_points[-1][0],
                inner_points[0][1] - inner_points[-1][1],
                inner_points[-1][0],
                inner_points[-1][1],
            )
            border_lines.append(final_outer_line)
            border_lines.append(final_inner_line)
            fill_in_line()
            save()
            finished = True

    screen.blit(fill_surf, (0, 0))
    for i in range(len(points)):
        if i > 0:
            pygame.draw.line(screen, "black", points[i - 1], points[i])
        pygame.draw.circle(screen, "black", points[i], 2)

    if not finished:
        if len(points):
            pygame.draw.line(screen, "red", points[-1], pos, 1)
    else:
        pygame.draw.line(screen, "black", points[-1], points[0])
        for line in border_lines:
            pygame.draw.line(
                screen,
                "black",
                (line[2], line[3]),
                (line[2] + line[0], line[3] + line[1]),
            )

    pygame.display.flip()
    clock.tick(60)
