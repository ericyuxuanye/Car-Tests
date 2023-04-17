# Car-Tests
Testing the performance of AI on a racetrack environment, using various different techniques including reinforcement learning and genetic algorithms.

## Setup

It's recommended you use a Python venv for this project. Run the following commands (assuming Anaconda):

```
conda create -n car-test python=3.9
conda activate car-test
pip install -r requirements.txt
```

to create and activate the conda environment and install all requirements. Note `python=3.9` is needed because some packages do not work with `python>=3.11`. To ensure the q_learning module works, modify line 14 of `model.py` to `device = "cuda"` or `device = "cpu"` if you are not on a Mac.

Each folder contains one model, which can be rendered using `python game.py`.