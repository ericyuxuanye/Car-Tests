import random
from itertools import count
import signal
import sys
from torch.multiprocessing import Pool, set_start_method
import numpy as np
from math import sin, cos, pi, atan2, sqrt
from functools import partial
from utils import (
    calc_distance_from_start,
    car_touching_line,
    get_distances,
    relative_car_velocities,
    line_angle,
)
from data import lines, border_lines, num_lines
from model import (
    create_net,
    crossover,
    mutate,
    get_population,
    set_params,
    Parameters,
    convert_to_tensor,
    tensor_to_action,
    save_model,
)

try:
    set_start_method("spawn")
except RuntimeError:
    pass


class Car:
    def __init__(self, friction, starting_index):
        super().__init__()
        self.width = 24
        self.height = 47
        self.friction = friction
        self.just_hit = False
        # idx = random.randint(0, len(lines)-1)
        line = lines[starting_index]
        self.x = line[2]
        self.y = line[3]
        self.velocity = np.array([0.0, 0.0])
        self.prev_distance = starting_index
        self.rotation = line_angle(line)

    def step(self, action):
        accel, steer = action[0], action[1]
        accel *= 2
        steer *= 10
        self.rotation += steer

        radians = self.rotation / 180 * pi + pi / 2
        self.velocity[0] += accel * cos(radians)
        # subtract because y is flipped
        self.velocity[1] -= accel * sin(radians)
        # friction calculation
        r = sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
        theta = atan2(self.velocity[1], self.velocity[0])
        r = max(r - self.friction, 0)
        self.velocity[0] = r * cos(theta)
        self.velocity[1] = r * sin(theta)

        self.x += self.velocity[0]
        self.y += self.velocity[1]
        just_hit = False
        if car_touching_line(
            self.x, self.y, self.width, self.height, self.rotation, lines, border_lines
        ):
            just_hit = True

        return self.get_state(), self.get_reward(), just_hit

    def get_state(self):
        state = np.empty((10,), dtype=np.float32)
        distances = get_distances((self.x, self.y), self.rotation, border_lines)
        state[:8] = distances
        velocities = relative_car_velocities(self.velocity, self.rotation)
        state[8:10] = velocities
        return state

    def get_reward(self):
        curr_distance = calc_distance_from_start((self.x, self.y), lines)
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
        return reward


# def select(pop: list[Parameters], fitnesses: np.ndarray) -> list[Parameters]:
#     """
#     Select a new population
# 
#     @params
#         pop (List[Parameters]): The entire population of parameters
#         fitnesses (np.ndarray): the fitnesses for each entity in the population
#     @returns
#         List[Parameters]: A new population made of fitter individuals
#     """
#     idx = np.random.choice(
#         np.arange(len(pop)),
#         size=len(pop),
#         replace=True,
#         p=fitnesses / (fitnesses.sum()),
#     )
#     return [pop[i] for i in idx]

def makeWheel(population, fitness: np.ndarray):
    wheel = []
    total = fitness.sum()
    top = 0
    for p, f in zip(population, fitness):
        f = f/total
        wheel.append((top, top+f, p))
        top += f
    return wheel

def binSearch(wheel, num):
    mid = len(wheel)//2
    low, high, answer = wheel[mid]
    if low<=num<=high:
        return answer
    elif high < num:
        return binSearch(wheel[mid+1:], num)
    else:
        return binSearch(wheel[:mid], num)

def select(wheel, N):
    stepSize = 1.0/N
    answer = []
    r = random.random()
    answer.append(binSearch(wheel, r))
    while len(answer) < N:
        r += stepSize
        if r>1:
            r %= 1
        answer.append(binSearch(wheel, r))
    return answer

def init_worker():
    global model
    model = create_net()

def fitness(params, starting_index):
    set_params(model, params)
    car = Car(1, starting_index)
    observation = car.get_state()
    total_reward = 0
    for _ in range(1200):
        action = tensor_to_action(model(convert_to_tensor(observation)))
        observation, reward, crashed = car.step(action)
        total_reward += reward
        if crashed:
            break
    return total_reward

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    population = get_population()
    print("Initial population: ", len(population))
    with Pool(8, initializer=init_worker) as pool:

        def handler(sig, frame):
            net = create_net()
            set_params(net, fittest)
            save_model(net, population)
            sys.exit(0)

        signal.signal(signal.SIGINT, handler)
        for i in count():
            # we start at a different location each generation
            starting_location = random.randint(0, num_lines - 1)
            # starting_location = 0
            fitness_eval = partial(fitness, starting_index=starting_location)
            fitnesses = np.array(pool.map(fitness_eval, population))
            fittest = population[fitnesses.argmax()]
            avg_fitness = fitnesses.sum() / len(fitnesses)
            n_fittest = [population[x] for x in np.argpartition(fitnesses, -10)[-10:]]
            wheel = makeWheel(population, np.clip(fitnesses, 1, None))
            population = select(wheel, len(population) - 10)
            # min_fitness = fitnesses.min()
            # population = select(population, np.clip(fitnesses, 1, None))
            # random.shuffle(population)
            # population = population[:-10]
            population.extend(n_fittest)
            pop2 = list(population)
            for j in range(len(population) - 10):
                child = crossover(population[j], pop2)
                child = mutate(child)
                population[j] = child
            print(f"Generation {i}. avg: {avg_fitness:6.2f}, fittest: {fitnesses.max():6.2f}")
