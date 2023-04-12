from __future__ import annotations
import numpy as np
import os
import torch
from torch import nn
from torch.nn.parameter import Parameter

device = "cpu"  # change to "cuda" for nvidia, "cpu" for cpu
# because we do not need gradients for GA
torch.set_grad_enabled(False)

Parameters = list[Parameter]

OBSERVATIONS = 10
ACTIONS = 9
POP_SIZE = 150

MUTATION_FACTOR = 0.003
MUTATION_RATE = 0.15
CROSS_RATE = 0.15


def create_net():
    return nn.Sequential(
        nn.Linear(OBSERVATIONS, 32, bias=True),
        nn.ReLU(),
        nn.Linear(32, 16, bias=True),
        nn.ReLU(),
        nn.Linear(16, ACTIONS, bias=True),
    ).to(device)


def convert_to_tensor(observation):
    return torch.tensor(observation).float().unsqueeze(0).to(device)


def tensor_to_action(tensor: torch.Tensor):
    # return torch.multinomial(tensor, 1, True).item()
    return torch.argmax(tensor.cpu()).item()


def get_params(net: torch.nn.Sequential) -> Parameters:
    """
    Gets the parameters from a PyTorch model stored as an nn.Sequential

    @params
        network (nn.Sequential): A pytorch model
    @returns
        Parameters: the parameters of the model
    """
    params = []
    for layer in net:
        if hasattr(layer, "weight") and layer.weight != None:
            params.append(layer.weight)
        if hasattr(layer, "bias") and layer.bias != None:
            params.append(layer.bias)
    return params


def set_params(net: torch.nn.Sequential, params: Parameters) -> torch.nn.Sequential:
    """
    Sets the parameters for an nn.Sequential

    @params
        network (torch.nn.Sequential): A network to change the parameters of
        params (Parameters): Parameters to place into the model
    @returns
        torch.nn.Sequential: A model the the provided parameters
    """
    i = 0
    for layerid, layer in enumerate(net):
        if hasattr(layer, "weight") and layer.weight != None:
            net[layerid].weight = params[i]
            i += 1
        if hasattr(layer, "bias") and layer.bias != None:
            net[layerid].bias = params[i]
            i += 1
    return net


def crossover(parent1: Parameters, pop: list[Parameters]) -> Parameters:
    """
    Crossover two individuals and produce a child.

    This is done by randomly splitting the weights and biases at each layer for the parents and then
    combining them to produce a child

    @params
        parent1 (Parameters): A parent that may potentially be crossed over
        pop (List[Parameters]): The population of solutions
    @returns
        Parameters: A child with attributes of both parents or the original parent1
    """
    if np.random.rand() < CROSS_RATE:
        i = np.random.randint(0, len(pop), size=1)[0]
        parent2 = pop[i]
        child = []
        split = np.random.rand()

        for p1l, p2l in zip(parent1, parent2):
            splitpoint = int(len(p1l) * split)
            new_param = nn.parameter.Parameter(
                torch.cat([p1l[:splitpoint], p2l[splitpoint:]])
            )
            child.append(new_param)

        return child
    else:
        return parent1


def gen_mutate(shape: torch.Size) -> torch.Tensor:
    """
    Generate a tensor to use for random mutation of a parameter

    @params
        shape (torch.Size): The shape of the tensor to be created
    @returns
        torch.tensor: a random tensor
    """
    return (
        nn.Dropout(MUTATION_RATE)(torch.ones(shape) - 0.5)
        * torch.randn(shape).to(device)
        * MUTATION_FACTOR
    )


def mutate(child: Parameters) -> Parameters:
    """
    Mutate a child

    @params
        child (Parameters): The original parameters
    @returns
        Parameters: The mutated child
    """
    for i in range(len(child)):
        for j in range(len(child[i])):
            child[i][j] += gen_mutate(child[i][j].shape)

    return child


def save_model(model: torch.nn.Module, population: list[Parameters]):
    torch.save(model.state_dict(), "best.pt")
    torch.save(population, "population.pt")


def load_model():
    net = create_net()
    net.load_state_dict(torch.load("best.pt"))
    return net

def get_population():
    if os.path.isfile("./population.pt"):
        return torch.load("./population.pt")
    net = create_net()
    base = get_params(net)
    shapes = [param.shape for param in base]
    pop = []
    for _ in range(POP_SIZE):
        entity = []
        for shape in shapes:
            # if fan in and fan out can be calculated (tensor is 2d) then using kaiming uniform initialisation
            # as per nn.Linear
            # otherwise use uniform initialisation between -0.5 and 0.5
            try:
                rand_tensor = nn.init.kaiming_uniform_(torch.empty(shape)).to(device)
            except ValueError:
                rand_tensor = nn.init.uniform_(torch.empty(shape), -0.2, 0.2).to(device)
            entity.append((torch.nn.parameter.Parameter(rand_tensor)))
        pop.append(entity)
    return pop
