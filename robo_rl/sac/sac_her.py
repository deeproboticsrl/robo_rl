import argparse
import os

import gym
import numpy as np
import torch
from robo_rl.common import Buffer
from robo_rl.common.utils import gym_torchify, print_heading
from robo_rl.sac import SAC, TanhSquasher
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch on fire')
parser.add_argument('--env_name', default="FetchReach-v1", help="Should be GoalEnv")
parser.add_argument('--env_seed', type=int, default=0, help="environment seed")
parser.add_argument('--soft_update_tau', type=float, default=0.005, help="target smoothening coefficient tau")
parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
parser.add_argument('--discount_factor', type=float, default=0.99, help='discount factor gamma')
parser.add_argument('--scale_reward', type=int, default=10,
                    help="reward scaling humannoid_v1=20, humnanoid_rllab=10, other mujoco=5")
parser.add_argument('--reparam', type=bool, default=True, help="True if reparameterization trick is applied")
parser.add_argument('--deterministic', type=bool, default=False)
parser.add_argument('--target_update_interval', type=int, default=1,
                    help="used in case of hard update")
parser.add_argument('--td3_update_interval', type=int, default=50,
                    help="used in case of delayed update for policy")

parser.add_argument('--hidden_dim', type=int, default=256, help='no of hidden units ')
parser.add_argument('--buffer_capacity', type=int, default=1000000, help='buffer capacity')
parser.add_argument('--sample_batch_size', type=int, default=256, help='number of samples from replay buffer')
parser.add_argument('--max_time_steps', type=int, default=10000, help='max number of env timesteps per episodes')
parser.add_argument('--num_episodes', type=int, default=101, help='number of episodes')
parser.add_argument('--updates_per_step', type=int, default=100, help='updates per step')
parser.add_argument('--save_iter', type=int, default=100, help='save model and buffer '
                                                               'after certain number of iteration')
args = parser.parse_args()

env = gym.make(args.env_name)

env.seed(args.env_seed)
torch.manual_seed(args.env_seed)
np.random.seed(args.env_seed)

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.spaces["observation"].shape[0]
goal_dim = env.observation_space.spaces["achieved_goal"].shape[0]
hidden_dim = [args.hidden_dim, args.hidden_dim]

unbiased = True
if unbiased:
    logdir = "./tensorboard_log/unbiased_her/"
else:
    logdir = "./tensorboard_log/biased_her/"

os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(log_dir=logdir)

squasher = TanhSquasher()

sac = SAC(action_dim=action_dim, state_dim=state_dim + goal_dim, hidden_dim=hidden_dim,
          discount_factor=args.discount_factor,
          writer=writer, scale_reward=args.scale_reward, reparam=args.reparam, deterministic=args.deterministic,
          target_update_interval=args.target_update_interval, lr=args.lr, soft_update_tau=args.soft_update_tau,
          td3_update_interval=args.td3_update_interval, squasher=squasher)

buffer = Buffer(capacity=args.buffer_capacity)
rewards = []
update_count = 0
max_accuracy = -np.inf


def ld_to_dl(batch_list_of_dicts):
    batch_dict_of_lists = {}
    for k in batch_list_of_dicts[0].keys():
        batch_dict_of_lists[k] = []
    for dictionary in batch_list_of_dicts:
        for k in batch_list_of_dicts[0].keys():
            batch_dict_of_lists[k].append(dictionary[k])
    return batch_dict_of_lists


# num episodes after which to test
test_interval = 20
num_tests = 50

for cur_episode in range(1, args.num_episodes+1):
    print(f"Starting episode {cur_episode}")

    reset_obs = env.reset()
    state = torch.Tensor(reset_obs["observation"])
    desired_goal = torch.Tensor(reset_obs["desired_goal"])
    done = False
    timestep = 0
    episode_buffer = []

    while not done and timestep <= args.max_time_steps:
        episode_reward = 0
        action, log_prob = sac.get_action(torch.cat([state, desired_goal]), evaluate=True)
        action = action.detach()
        observation, reward, done, _ = gym_torchify(env.step(action.numpy()), is_goal_env=True)
        episode_buffer.append(dict(state=state, next_state=observation["observation"], action=action,
                                   done=done, achieved_goal=observation["achieved_goal"], log_prob=log_prob))
        sample = dict(state=torch.cat([state, desired_goal]), action=action, reward=reward,
                      next_state=torch.cat([observation["observation"], desired_goal]), done=done)
        buffer.add(sample)

        episode_reward += reward
        state = observation["observation"]
        timestep += 1

    # add hindsight transitions
    for transition in episode_buffer:

        final_goal = observation["achieved_goal"]
        state = torch.cat([transition["state"], final_goal])
        log_prob_final_goal = sac.policy.compute_log_prob_action(state, sac.squasher, transition["action"]).detach()

        reward = env.compute_reward(achieved_goal=transition["achieved_goal"], desired_goal=final_goal, info=None)
        unbiased_reward = torch.Tensor(reward * np.exp(log_prob_final_goal) / np.exp(transition["log_prob"].detach()))

        if unbiased:
            reward = unbiased_reward
        hindisght_sample = dict(state=state,
                                action=transition["action"],
                                reward=torch.Tensor([reward]).detach(),
                                done=transition["done"],
                                next_state=torch.cat([transition["next_state"], final_goal]))
        buffer.add(hindisght_sample)

    if len(buffer) > args.sample_batch_size:
        for num_update in range(args.updates_per_step):
            update_count += 1
            batch_list_of_dicts = buffer.sample(batch_size=args.sample_batch_size)
            batch_dict_of_lists = ld_to_dl(batch_list_of_dicts)
            sac.policy_update(batch_dict_of_lists, update_number=update_count)

    if cur_episode % test_interval == 0:
        # test
        successes = []
        for i in range(num_tests):
            reset_obs = env.reset()
            state = torch.Tensor(reset_obs["observation"])
            desired_goal = torch.Tensor(reset_obs["desired_goal"])
            done = False
            timestep = 0

            success = False
            while not done and timestep <= args.max_time_steps:
                action = sac.get_action(torch.cat([state, desired_goal]),deterministic=True).detach()
                observation, reward, done, info = gym_torchify(env.step(action.numpy()), is_goal_env=True)
                state = observation["observation"]
                timestep += 1
                if 'is_success' in info:
                    success = info['is_success']
            successes.append(success)

        accuracy = sum(successes) / num_tests
        print_heading(f"Finished episode {cur_episode}")
        print(f"Accuracy {accuracy}")

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            # save current best model
            print(f"\nNew best model with accuracy {max_accuracy}")
            sac.save_model(env_name=args.env_name, info='best')

        sac.writer.add_scalar("Accuracy ", accuracy, cur_episode/test_interval)

    if cur_episode % args.save_iter == 0:
        print(f"\nSaving periodically - iteration {cur_episode}")
        sac.save_model(env_name=args.env_name, info=str(cur_episode))
        buffer.save_buffer(info=args.env_name)
