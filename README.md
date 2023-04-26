# Car-Tests

Testing the performance of AI on a racetrack environment, using various
different techniques including reinforcement learning and genetic algorithms.

## Prerequisites:

Install Python: https://www.python.org/downloads/

Install Anaconda: https://www.anaconda.com/download/

## Setup

It's recommended you use a Python venv for this project. Run the following
commands (assuming Anaconda):

```
conda create -n car-test python=3.9
conda activate car-test
pip install -r requirements.txt
```

to create and activate the conda environment and install all requirements. Note
`python=3.9` is needed because some packages do not work with `python>=3.11`.

Each folder contains one model, which can be rendered using `python game.py`.

## Notes on the environment

The environment for the subprojects are mostly the same. They all use a
continuous state space, consisting of the distance of the walls from the car in
8 directions and the velocity of the car in the forward/backward and left/right
directions. However, some of the environments have continuous action spaces
while others have discrete action spaces. In the case of a continuous action
space, the car only outputs two numbers: the amount to accelerate, and the
amount to steer. In the case of a discrete action space, the car outputs the
estimated reward for performing one of 10 different actions, which are all
combinations of accelerating forward/backward, and steering left/right.
