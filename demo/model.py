import math
import random
import os
from collections import namedtuple, deque
from itertools import count
from utils import get_distances, lines

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = "cpu"

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        return self.layer3(x)


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 400000
TAU = 0.01
LR = 1e-4

n_actions = 9
n_observations = 10

policy_net = DQN(n_observations, n_actions).to(device)
if os.path.exists("policy_net.pt"):
    policy_net.load_state_dict(torch.load("policy_net.pt"))
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR, amsgrad=True)

memory = ReplayMemory(10000)

steps_done = 0
if os.path.exists("steps.txt"):
    with open("steps.txt", "rt") as f:
        steps_done = int(f.readline())


def select_action(state):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    if sample > eps_threshold:
        with torch.no_grad():
            # pick best action
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # pick random action
        return torch.tensor([[random.randint(0, n_actions-1)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose states
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    # Compute mask of non-final states and concatenate batch elements
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1})
    # with default punishment
    next_state_values = torch.full((BATCH_SIZE,), -10, device=device, dtype=torch.float)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # Compute expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # In place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


prev_state = None
prev_action = None


def get_action(state):
    global prev_state, prev_action
    state = torch.FloatTensor(state).reshape((1, 10)).to(device)
    prev_state = state
    action = select_action(state)
    prev_action = action
    return action[0, 0].int()


def update_model(new_state, reward, just_hit):
    global steps_done
    steps_done += 1
    if just_hit:
        new_state = None
    else:
        new_state = torch.FloatTensor(new_state).reshape((1, 10)).to(device)
    reward = torch.tensor([reward], device=device)
    memory.push(prev_state, prev_action, new_state, reward)
    optimize_model()

    # soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[
            key
        ] * TAU + target_net_state_dict[key] * (1 - TAU)
    target_net.load_state_dict(target_net_state_dict)


def save_model():
    torch.save(policy_net.state_dict(), f="policy_net.pt")
    with open("steps.txt", "wt") as f:
        print(steps_done, file=f)
