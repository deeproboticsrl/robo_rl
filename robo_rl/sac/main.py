import argparse

import gym
import numpy as np
import torch
from osim.env import ProstheticsEnv
from robo_rl.sac.softactorcritic import SAC
from robo_rl.common.replaybuffer import ReplayBuffer

parser = argparse.ArgumentParser(description='PyTorch on fire')
parser.add_argument('--env_name', default="ProstheticsEnv")
parser.add_argument('--scale_reward', type=int, default=3)
parser.add_argument('--env_seed', type=int, default=1105, help="environment seed")
parser.add_argument('--soft_update_tau', type=float, default=0.005, help="target smoothening coefficient tau")
parser.add_argument('--discount_factor', type=float, default=0.99, help='discount factor gamma')
parser.add_argument('--scale_reward', type=int, default=10,
                    help="reward scaling humannoid_v1=20, humnanoid_rllib=10, other mujoco=5")
parser.add_argument('--reparam', type=bool, default=True, help="True if reparameterization trick is applied")
parser.add_argument('--target_update_interval', type=int, default=1,
                    help='used in case of hard update with or without td3')
parser.add_argument('--hidden_dim', type=int, default=256, help='no of hidden units ')
parser.add_argument('--buffercapacity', type=int, default=1000000, help='buffer capacity')
parser.add_argument('--sample_batch_size', type=int, default=256, help='sample from replay buffer')
args = parser.parse_args()
if args.env_name == "ProstheticsEnv":
    env = ProstheticsEnv()

else:
    ## Need to normalize the actions
    env = gym.make(args.env_name)

env.seed(args.env_seed)
torch.manual_seed(args.env_seed)
np.random.seed(args.env_seed)

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]

writer = SummaryWriter()
sacobj = SAC(action_dim, state_dim, args.hidden_dim, args.discount_factor, args.scale_reward, args.reparam,
             args.deterministic, args.target_update_interval, args.lr, args.soft_update_tau, args.td3)

buffer = ReplayBuffer(capacity=args.buffercapacity)
num_iteration = 10000

for i in range(num_iteration):
    state = env.reset()
    action = sacobj.policy.get_action(state)
    observation, reward, done, _ = env.step(action)
    sample = dict(state=state, action=action, reward=reward, next_state=observation, done=done)
    buffer.add(sample)
    if i > args.sample_batch_size:

        batch = buffer.sample(batch_size=args.sample_batch_size)
        sacobj.policy_update(batch, update_number=i)
