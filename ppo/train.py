import random
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
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gym import Env, spaces

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


class Car(Env):
    def __init__(self, acceleration, friction, rot_speed):
        super().__init__()
        self.width = 24
        self.height = 47
        self.rot_speed = rot_speed
        self.accel = acceleration
        self.friction = friction
        self.just_hit = False
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)
        self.episode = 0

    def reset(self):
        idx = random.randint(0, len(lines)-1)
        line = lines[idx]
        self.x = line[2]
        self.y = line[3]
        self.velocity = [0.0, 0.0]
        self.prev_distance = idx
        self.rotation = line_angle(line)
        return self.get_state()

    def step(self, action):
        left, right, forward, backward = action_to_keys[action]
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
            self.episode += 1
            just_hit = True

        return self.get_state(), self.get_reward(just_hit), just_hit, {}

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


env = Car(1.5, 1, 7)
check_env(env)

model = PPO("MlpPolicy", env, verbose=2)
# model = PPO.load("ppo_racer", env=env)
model.learn(total_timesteps=200_000)
model.save("ppo_racer")
