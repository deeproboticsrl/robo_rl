import os
import pickle
import gym
import numpy as np
import torch
from osim.env import ProstheticsEnv
from robo_rl.common import Buffer
from robo_rl.sac import SAC, SigmoidSquasher
from tensorboardX import SummaryWriter
from robo_rl.common.utils import gym_torchify
from robo_rl.sac import get_sac_parser
from torch.optim import Adam, SGD
import torch.nn.functional as f

parser = get_sac_parser()
parser.add_argument('--env_name', default="Reacher-v2")
parser.add_argument('--mode', default='train')

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
logdir += f"{args.env_name}_lr={args.lr}_rs={args.scale_reward}_tau={args.discount_factor}_activation=relu/_old_policy"
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(log_dir=logdir)

squasher = SigmoidSquasher()

sac = SAC(action_dim=action_dim, state_dim=state_dim, hidden_dim=hidden_dim, discount_factor=args.discount_factor,
          writer=writer, optimizer=Adam, scale_reward=args.scale_reward, reparam=args.reparam,
          deterministic=args.deterministic, activation=f.relu,
          target_update_interval=args.target_update_interval, lr=args.lr, soft_update_tau=args.soft_update_tau,
          td3_update_interval=args.td3_update_interval, squasher=squasher)

buffer = Buffer(capacity=args.buffer_capacity)


def ld_to_dl(batch_list_of_dicts):
    batch_dict_of_lists = {}
    for k in batch_list_of_dicts[0].keys():
        batch_dict_of_lists[k] = []
    for dictionary in batch_list_of_dicts:
        for k in batch_list_of_dicts[0].keys():
            batch_dict_of_lists[k].append(dictionary[k])
    return batch_dict_of_lists


def train():
    rewards = []
    update_count = 1
    max_reward = -np.inf

    for cur_episode in range(args.num_episodes):
        print(f"Starting episode {cur_episode}")

        state = torch.Tensor(env.reset())
        done = False
        timestep = 0
        episode_reward = 0

        while not done and timestep <= args.max_time_steps:
            action = sac.get_action(state, deterministic=True).detach()
            observation, reward, done, _ = gym_torchify(env.step(action))
            sample = dict(state=state, action=action, reward=reward, next_state=observation, done=done)
            buffer.add(sample)
            if len(buffer) > 3 * args.sample_batch_size:
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
            episode_avg_reward = float(episode_reward) / timestep
        if episode_reward > max_reward:
            max_avg_reward = episode_avg_reward
            # save current best model
            print(f"\nNew best model with reward {max_avg_reward}")
            sac.save_model(env_name=args.env_name, info=f'best')

        if cur_episode % args.save_iter == 0:
            print(f"\nSaving periodically - iteration {cur_episode}")
            sac.save_model(env_name=args.env_name, info=str(cur_episode))
            buffer.save_buffer(info=args.env_name)

        sac.writer.add_scalar("Episode average Reward", episode_avg_reward, cur_episode)
        rewards.append(episode_reward)
        print(rewards)


def play(env):
    actor_path = f'model/{args.env_name}/actor_best.pt'

    sac.load_model(actor_path)
    print(args.env_name)
    for i in range(100):
        observation, reward, done, _ = env.step(sac.get_action(deterministic=True))
        print(args.env_name)
        env.render()


if args.mode == 'train':
    train()
elif args.mode is 'play':
    play(env)
