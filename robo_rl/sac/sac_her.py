import os

import gym
import numpy as np
import torch
from robo_rl.common import Buffer
from robo_rl.common.utils import gym_torchify, print_heading
from robo_rl.sac import SAC, TanhSquasher
from robo_rl.sac import get_sac_parser, get_logfile_name
from tensorboardX import SummaryWriter
from torch.optim import Adam

optimizer = Adam

parser = get_sac_parser()
parser.add_argument('--env_name', default="FetchReach-v1", help="Should be GoalEnv")
parser.add_argument('--distance_threshold', type=float, default=0.01, help='Threshold for success in binary reward')
args = parser.parse_args()

env = gym.make(args.env_name)
env.distance_threshold = args.distance_threshold

env.seed(args.env_seed)
torch.manual_seed(args.env_seed)
np.random.seed(args.env_seed)

logdir = "./tensorboard_log/"
# logdir += "dummy"
modeldir = f"./model/{args.env_name}/"
bufferdir = f"./buffer/{args.env_name}"

logfile = get_logfile_name(args)

action_dim = env.action_space.shape[0]
if args.goal_obs:
    state_dim = env.observation_space.spaces["achieved_goal"].shape[0]
else:
    state_dim = env.observation_space.spaces["observation"].shape[0]
goal_dim = env.observation_space.spaces["achieved_goal"].shape[0]
hidden_dim = [args.hidden_dim] * 2

os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(log_dir=logdir + logfile)

squasher = TanhSquasher()

sac = SAC(action_dim=action_dim, state_dim=state_dim + goal_dim, hidden_dim=hidden_dim,
          discount_factor=args.discount_factor, optimizer=optimizer, policy_lr=args.policy_lr, critic_lr=args.critic_lr,
          value_lr=args.value_lr, writer=writer, scale_reward=args.scale_reward, reparam=args.reparam,
          target_update_interval=args.target_update_interval, soft_update_tau=args.soft_update_tau,
          td3_update_interval=args.td3_update_interval, squasher=squasher, policy_weight_decay=args.policy_weight_decay,
          critic_weight_decay=args.critic_weight_decay, value_weight_decay=args.value_weight_decay,
          grad_clip=args.grad_clip, loss_clip=args.loss_clip, clip_val_grad=args.clip_val_grad,
          deterministic=args.deterministic, clip_val_loss=args.clip_val_loss, log_std_min=args.log_std_min,
          log_std_max=args.log_std_max)

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


for cur_episode in range(1, args.num_episodes + 1):
    print(f"Starting episode {cur_episode}")

    reset_obs = env.reset()
    if args.goal_obs:
        state = torch.Tensor(reset_obs["achieved_goal"])
    else:
        state = torch.Tensor(reset_obs["observation"])
    desired_goal = torch.Tensor(reset_obs["desired_goal"])
    done = False
    timestep = 0
    episode_buffer = []

    while not done and timestep <= args.max_time_steps:
        episode_reward = 0
        action, log_prob = sac.get_action(torch.cat([state, desired_goal]), evaluate=True,
                                          deterministic=args.deterministic)
        action = action.detach()
        observation, reward, done, _ = gym_torchify(env.step(action.numpy()), is_goal_env=True)
        if args.positive_reward:
            reward = reward + 1
        if args.goal_obs:
            next_state = observation["achieved_goal"]
        else:
            next_state = observation["observation"]
        episode_buffer.append(dict(state=state, next_state=next_state, action=action,
                                   done=done, achieved_goal=observation["achieved_goal"], log_prob=log_prob))
        sample = dict(state=torch.cat([state, desired_goal]), action=action, reward=reward,
                      next_state=torch.cat([next_state, desired_goal]), done=done)
        buffer.add(sample)

        episode_reward += reward
        state = next_state
        timestep += 1

    sac.writer.add_scalar("Policy linear layer 1 weight 0", sac.policy.linear_layers[0].weight[0][0], cur_episode)
    for name, param in sac.policy.named_parameters():
        sac.writer.add_histogram("policy_" + name, param, cur_episode)
    for name, param in sac.value.named_parameters():
        sac.writer.add_histogram("value_" + name, param, cur_episode)
    for name, param in sac.critics[0].named_parameters():
        sac.writer.add_histogram("critic1_" + name, param, cur_episode)
    for name, param in sac.critics[1].named_parameters():
        sac.writer.add_histogram("critic2_" + name, param, cur_episode)

    # add hindsight transitions
    if args.goal_obs:
        final_goal = observation["achieved_goal"]
    else:
        final_goal = observation["observation"]

    for transition in episode_buffer:
        if args.rewarding:
            final_goal = transition["achieved_goal"]
        state = torch.cat([transition["state"], final_goal])
        log_prob_final_goal = sac.policy.compute_log_prob_action(state, sac.squasher, transition["action"]).detach()

        reward = env.compute_reward(achieved_goal=transition["achieved_goal"], desired_goal=final_goal, info=None)
        if args.positive_reward:
            reward = reward + 1
        unbiased_reward = torch.Tensor(reward * np.exp(log_prob_final_goal) / np.exp(transition["log_prob"].detach()))

        if args.unbiased:
            reward = unbiased_reward
        done = transition["done"]
        if args.rewarding:
            done = torch.Tensor([True])
        hindisght_sample = dict(state=state,
                                action=transition["action"],
                                reward=torch.Tensor([reward]).detach(),
                                done=done,
                                next_state=torch.cat([transition["next_state"], final_goal]))
        buffer.add(hindisght_sample)

    if len(buffer) > args.sample_batch_size:
        for num_update in range(args.updates_per_step):
            update_count += 1
            batch_list_of_dicts = buffer.sample(batch_size=args.sample_batch_size)
            batch_dict_of_lists = ld_to_dl(batch_list_of_dicts)
            sac.policy_update(batch_dict_of_lists, update_number=update_count)

    if cur_episode % args.test_interval == 0:
        # test
        successes = []
        for i in range(args.num_tests):
            reset_obs = env.reset()
            if args.goal_obs:
                state = torch.Tensor(reset_obs["achieved_goal"])
            else:
                state = torch.Tensor(reset_obs["observation"])
            desired_goal = torch.Tensor(reset_obs["desired_goal"])
            done = False
            timestep = 0

            success = False
            while not done and timestep <= args.max_time_steps:
                action = sac.get_action(torch.cat([state, desired_goal]), deterministic=True).detach()
                observation, reward, done, info = gym_torchify(env.step(action.numpy()), is_goal_env=True)
                if args.goal_obs:
                    state = observation["achieved_goal"]
                else:
                    state = observation["observation"]

                timestep += 1
                if 'is_success' in info:
                    success = info['is_success']
            successes.append(success)

        accuracy = sum(successes) / args.num_tests
        print_heading(f"Finished episode {cur_episode}")
        print(f"Accuracy {accuracy}")

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            # save current best model
            print(f"\nNew best model with accuracy {max_accuracy}")
            sac.save_model(all_nets_path=modeldir + logfile + "/", env_name=args.env_name, info='best')

        sac.writer.add_scalar("Accuracy ", accuracy, cur_episode / args.test_interval)

    if cur_episode % args.save_iter == 0:
        print(f"\nSaving periodically - iteration {cur_episode}")
        sac.save_model(all_nets_path=modeldir + logfile + "/", env_name=args.env_name, info="periodic")
        buffer.save_buffer(path=bufferdir + logfile + "/", info=args.env_name)
