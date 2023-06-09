from neat.math_util import softmax
import pickle
import numpy as np
from math import sin, cos, pi, atan2, sqrt
from utils import (
    calc_distance_from_start,
    car_touching_line,
    get_distances,
    relative_car_velocities,
    line_angle,
)
from data import num_lines, border_lines, lines
import neat
import os

class Car():
    def __init__(self, friction):
        super().__init__()
        self.width = 24
        self.height = 47
        self.friction = friction
        self.just_hit = False
        # idx = random.randint(0, len(lines)-1)
        idx = 13
        line = lines[idx]
        self.x = line[2]
        self.y = line[3]
        self.velocity = np.array([0.0, 0.0])
        self.prev_distance = idx
        self.rotation = line_angle(line)

    def step(self, action):
        accel, steer = action
        accel = np.tanh(accel) * 5
        steer = np.tanh(steer) * 10
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
        if car_touching_line(self.x, self.y, self.width, self.height, self.rotation, lines, border_lines):
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
        return reward


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    car = Car(1)
    observation = car.get_state()
    total_reward = 0
    for _ in range(1200):
        action = net.activate(observation)
        observation, reward, crashed = car.step(action)
        total_reward += reward
        if crashed:
            break
    return total_reward


def run(config_file):
    evaluator = neat.ParallelEvaluator(8, eval_genome)
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(evaluator.evaluate, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    print('Fitness:', winner.fitness)
    with open("neat_genome", "wb") as f:
        pickle.dump(winner, f)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
