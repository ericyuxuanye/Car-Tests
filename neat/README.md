# Neuroevolution Through Augmented Topologies

This implements the Neuroevolution Through Augmented Topologies algorithm
(NEAT). This algorithm is a type of genetic algorithm that evolves neural
networks, which requires no gradient calculations. One distinction from genetic
algorithm is that the hyperparameters of neural networks are also controlled by
NEAT, and they get more complex as they evolve. The environment used in this case
uses a continuous action space because this allows smaller neural networks.

## Training

To train the agent, first make sure that the `config-feedforward` has the
configuration that is desired. Additionally, modify `train.py` so that the
number of generations is correct. Then you can do `python train.py`.

## Running

To run the agent, run `python game.py`.
