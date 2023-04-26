# Proximal Policy Optimization

Implements Proximal Policy Optimization with a discrete action space.
Proximal Policy Optimization is a type of policy gradient Reinforcement learning algorithm, that either uses a negative reward or a clipped gradient in order to make sure that each update does not stray from the current policy too much.
Just a normal Actor Critic algorithm, Proximal Policy Optimization uses both a policy network that tries to predict the action and a target network that tries to predict the value of being in a certain state.

## Training

To train the agent, make sure to modify `train.py` so that a sane number
of time steps is used. Then you can do `python train.py`.

## Running

To run the agent, run `python game.py`.
