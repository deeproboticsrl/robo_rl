import click
import gym
import numpy as np
from osim.env import ProstheticsEnv


@click.command()
@click.option('--env-name', default="ProstheticsEnv")
@click.option('--scale_reward', type=int,  default=3)
@click.option('--env_seed', type=int, default=1105, help="environment seed" )




if env-name=="ProstheticsEnv":
    env = ProstheticsEnv()

else:
    ## Need to normalize the actions
    env= gym.make(env-name)

env.seed(env_seed)
torch.manual_seed(env_seed)
np.random.seed(env_seed)

action_dim= env.action_space.shape[0]
state_dim = env.observation_space.shape[0]


