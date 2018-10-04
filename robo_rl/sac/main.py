import torch

import click
import gym
import numpy as np
from osim.env import ProstheticsEnv


@click.command()
@click.option('--env-name', default="ProstheticsEnv")
@click.option('--scale_reward', type=int, default=3)
@click.option('--env_seed', type=int, default=1105, help="environment seed")
@click.option('--soft_update_tau', type=float, default=0.005, help="target smoothening coefficient tau")
@click.option('--discount_factor', type=float, default=0.99, help='discount factor gamma')
@click.option('--scale_reward', type=int, default=10,
              help="reward scaling humannoid_v1=20, humnanoid_rllib=10, other mujoco=5")
@click.option('--reparam', type=bool, default=True, help="True if reparameterization trick is applied")
@click.option('--target_update_interval', type=int, default=1,
              help='used in case of hard update with or without td3')


if env-name == "ProstheticsEnv":
    env = ProstheticsEnv()

else:
    ## Need to normalize the actions
    env = gym.make(env - name)

env.seed(env_seed)
torch.manual_seed(env_seed)
np.random.seed(env_seed)

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
