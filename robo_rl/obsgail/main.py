import argparse
import os

import gym
import numpy as np
import torch
from osim.env import ProstheticsEnv
from robo_rl.common.networks import LinearDiscriminator
from robo_rl.obsgail import ExpertBuffer, ObsGAIL
from robo_rl.sac import SAC, SigmoidSquasher
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch on fire')
parser.add_argument('--env_name', default="Reacher-v2")
parser.add_argument('--env_seed', type=int, default=1105, help="environment seed")
parser.add_argument('--soft_update_tau', type=float, default=0.005, help="target smoothening coefficient tau")
parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
parser.add_argument('--discount_factor', type=float, default=0.99, help='discount factor gamma')
parser.add_argument('--scale_reward', type=int, default=10,
                    help="reward scaling humannoid_v1=20, humnanoid_rllab=10, other mujoco=5")
parser.add_argument('--reparam', type=bool, default=True, help="True if reparameterization trick is applied")
parser.add_argument('--deterministic', type=bool, default=False)
parser.add_argument('--target_update_interval', type=int, default=1,
                    help="used in case of hard update with or without td3")
parser.add_argument('--td3_update_interval', type=int, default=100,
                    help="used in case of delayed update for policy")

parser.add_argument('--hidden_dim', type=int, default=256, help='no of hidden units ')
parser.add_argument('--buffer_capacity', type=int, default=1000000, help='buffer capacity')
parser.add_argument('--sample_batch_size', type=int, default=256, help='number of samples from replay buffer')
parser.add_argument('--max_time_steps', type=int, default=10000, help='max number of env timesteps per episodes')
parser.add_argument('--num_episodes', type=int, default=10000, help='number of episodes')
parser.add_argument('--updates_per_step', type=int, default=1, help='updates per step')
parser.add_argument('--save_iter', type=int, default=100, help='save model and buffer '
                                                               'after certain number of iteration')
args = parser.parse_args()
if args.env_name == "ProstheticsEnv":
    env = ProstheticsEnv()

else:
    # Need to normalize the action_space
    env = gym.make(args.env_name)

env.seed(args.env_seed)
torch.manual_seed(args.env_seed)
np.random.seed(args.env_seed)

# TODO make sure state_dim action_dim works correctly for all kinds of envs.

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
hidden_dim = [args.hidden_dim, args.hidden_dim]

logdir = "./tensorboard_log/"
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(log_dir=logdir)

squasher = SigmoidSquasher()

sac = SAC(action_dim=action_dim, state_dim=state_dim, hidden_dim=hidden_dim, discount_factor=args.discount_factor,
          writer=writer, scale_reward=args.scale_reward, reparam=args.reparam, deterministic=args.deterministic,
          target_update_interval=args.target_update_interval, lr=args.lr, soft_update_tau=args.soft_update_tau,
          td3_update_interval=args.td3_update_interval, squasher=squasher)

# TODO use argparse

expert_buffer_capacity = 1000
expert_buffer = ExpertBuffer()

# TODO use proper path
expert_file_path = "experts/"

expert_buffer.add_from_file(expert_file_path=expert_file_path)

# TODO take from env
discriminator_input_dim = 1
discriminator_hidden_dim = [2, 3]

# TODO use VAE
discriminator = LinearDiscriminator(input_dim=discriminator_input_dim, hidden_dim=discriminator_hidden_dim)

obsgail = ObsGAIL(expert_buffer=expert_buffer, discriminator=discriminator, off_policy_algo=sac)

# TODO get from argparse
# obsgail.train(num_iterations=,learning_rate=,learning_rate_decay=,learning_rate_decay_training_steps=)

# TODO Gradient clipping in actor net

# TODO For SAC use reparam trick with normalising flow(??)

# TODO regularisation in form of gradient penalties for stable learning. makes GAN stable. Refer paper
# TODO Should we use simple weight regularisation then?
