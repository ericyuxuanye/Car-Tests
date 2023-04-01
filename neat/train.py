from neat.math_util import softmax
import pickle
import numpy as np
from math import sin, cos, pi, atan2, sqrt
from utils import (
    calc_distance_from_start,
    car_touching_line,
    get_distances,
    relative_car_velocities,
    num_lines,
    line_angle,
    lines,
)
import neat
import os
action_to_keys = [
    (0, 0, 0, 0),
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
    (1, 0, 0, 1),
    (1, 0, 1, 0),
    (0, 1, 1, 0),
    (0, 1, 0, 1),
]


class Car():
    def __init__(self, acceleration, friction, rot_speed):
        super().__init__()
        self.width = 24
        self.height = 47
        self.rot_speed = rot_speed
        self.accel = acceleration
        self.friction = friction
        self.just_hit = False
        # idx = random.randint(0, len(lines)-1)
        line = lines[0]
        self.x = line[2]
        self.y = line[3]
        self.velocity = [0.0, 0.0]
        self.prev_distance = 0
        self.rotation = line_angle(line)

    def step(self, action):
        probs = softmax(action)
        left, right, forward, backward = action_to_keys[np.random.choice(np.arange(8), p=probs)]
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
        just_hit = False
        if car_touching_line(self.x, self.y, self.width, self.height, self.rotation):
            just_hit = True

        return self.get_state(), self.get_reward(just_hit), just_hit

    def get_state(self):
        state = np.empty((10,), dtype=np.float32)
        distances = get_distances((self.x, self.y), self.rotation)
        state[:8] = distances
        velocities = relative_car_velocities(self.velocity, self.rotation)
        state[8:10] = velocities
        return state

    def get_reward(self, just_hit):
        if just_hit:
            return -10
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
        # if reward <= 0:
        #     # So that the agent does not stay in place
        #     reward = -0.1
        # So that agent does not stay in place
        # reward -= 0.1
        return reward


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    car = Car(1.5, 1, 7)
    observation = car.get_state()
    total_reward = 0
    while True:
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
    winner = p.run(evaluator.evaluate, 10)

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
