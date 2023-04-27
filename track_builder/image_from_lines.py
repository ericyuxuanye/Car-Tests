import numpy as np
from PIL import Image


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
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = c * f - a * h
        t = -(d * f - b * h + g * h - f * j) / denom
        s = (c * t + d - j) / h
    res = (denom != 0) & (t >= 0) & (0 <= s) & (s <= 1)
    return res


def get_color_array(image_array, border_lines):
    intersect_count = np.zeros((1024, 768), dtype=np.uint8)
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
    mask = intersect_count % 2 == 1
    image_array[mask, 0] = 46
    image_array[mask, 1] = 44
    image_array[mask, 2] = 42
    return image_array


def read_file(filename):
    with open(filename, "rt") as f:
        n = int(f.readline())
        lines = np.empty((n, 4))
        for i in range(n):
            lines[i] = tuple(map(int, f.readline().split()))
        border_lines = np.empty((n * 2, 4))
        for i in range(n * 2):
            border_lines[i] = tuple(map(float, f.readline().split()))

    return lines, border_lines


if __name__ == "__main__":
    lines, border_lines = read_file("lines.txt")
    background = Image.open("./background.jpg").resize((1024, 768))
    image_array = np.array(background).swapaxes(0, 1)
    color_array = get_color_array(image_array, border_lines)
    img = Image.fromarray(color_array.swapaxes(0, 1))
    img.save("background_with_track.png")
