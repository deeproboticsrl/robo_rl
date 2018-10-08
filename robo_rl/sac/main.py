import argparse
from tensorboardX import SummaryWriter
import gym
import numpy as np
import torch
from osim.env import ProstheticsEnv
from robo_rl.common import Buffer
from robo_rl.sac import SAC, SigmoidSquasher
import os

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
parser.add_argument('--sample_batch_size', type=int, default=256, help='sample from replay buffer')
parser.add_argument('--max_time_steps', type=int, default=10000, help='max number of env timesteps per episodes')
parser.add_argument('--num_episodes', type=int, default=10000, help='number of episodes')
parser.add_argument('--updates_per_step', type=int, default=1, help='updates per step')
parser.add_argument('--save_iter', type=int, default=10000, help='save model and buffer '
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

buffer = Buffer(capacity=args.buffer_capacity)
rewards = []
update_count = 0
max_reward = -np.inf


def gym_torchify(gym_out):
    observation, reward, done, info = gym_out
    return torch.Tensor(observation), torch.Tensor([reward]), torch.Tensor([done]), info


def ld_to_dl(batch_list_of_dicts):
    batch_dict_of_lists = {}
    for k in batch_list_of_dicts[0].keys():
        batch_dict_of_lists[k] = []
    for dictionary in batch_list_of_dicts:
        for k in batch_list_of_dicts[0].keys():
            batch_dict_of_lists[k].append(dictionary[k])
    return batch_dict_of_lists


for cur_episode in range(args.num_episodes):
    print(cur_episode)

    state = torch.Tensor(env.reset())
    done = False
    timestep = 0

    while not done and timestep <= args.max_time_steps:
        episode_reward = 0
        action = sac.get_action(state).detach()
        observation, reward, done, _ = gym_torchify(env.step(action))
        sample = dict(state=state, action=action, reward=reward, next_state=observation,done=done)
        buffer.add(sample)
        if len(buffer) > 10 * args.sample_batch_size:
            for num_update in range(args.updates_per_step):
                update_count += 1
                batch_list_of_dicts = buffer.sample(batch_size=args.sample_batch_size)
                batch_dict_of_lists = ld_to_dl(batch_list_of_dicts)

                """ Combined Experience replay. Add online transition too.
                """
                for k in batch_list_of_dicts[0].keys():
                    batch_dict_of_lists[k].append(sample[k])
                sac.policy_update(batch_dict_of_lists, update_number=update_count)

        episode_reward += reward
        state = observation
        timestep += 1

    if episode_reward > max_reward:
        max_reward = episode_reward
        # save current best model
        print(f"\nNew best model with reward {max_reward}")
        sac.save_model(env_name=args.env_name, info='best')

    if cur_episode % args.save_iter == 0:
        print(f"\nSaving periodically - iteration {cur_episode}")
        sac.save_model(env_name=args.env_name, info=str(cur_episode))
        buffer.save_buffer(info=args.env_name)

    sac.writer.add_scalar("Episode Reward", episode_reward)
    rewards.append(episode_reward)


