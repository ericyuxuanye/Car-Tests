# Car-Tests
Testing the performance of AI on a racetrack environment, using various different techniques including reinforcement learning and genetic algorithms.

## Prerequisites:

Install Python: https://www.python.org/downloads/

Install Anaconda: https://www.anaconda.com/download/

## Setup

It's recommended you use a Python venv for this project. Run the following commands (assuming Anaconda):

```
conda create -n car-test python=3.9
conda activate car-test
pip install -r requirements.txt
```

to create and activate the conda environment and install all requirements. Note `python=3.9` is needed because some packages do not work with `python>=3.11`.

Each folder contains one model, which can be rendered using `python game.py`.
