# NN Genetic Algorithm

This is an implementation of neuroevolution that only
modifies the parameters (not the topology, which NEAT implements).
The action space for the environment in this case is continuous because
this allows for simpler neural networks.

## Training

To train the agent, just run `python train.py`. You can exit out any time, using `SIGINT`,
and the data will be saved.

## Running

To run the agent, run `python game.py`.
