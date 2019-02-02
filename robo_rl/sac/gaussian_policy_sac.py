import os

import gym
import numpy as np
import torch
from osim.env import ProstheticsEnv
from robo_rl.common import Buffer
from robo_rl.common.utils import gym_torchify
from robo_rl.sac import SAC, TanhSquasher, GaussianPolicy
from robo_rl.sac import get_sac_parser, get_logfile_name
from tensorboardX import SummaryWriter
from torch.optim import Adam

optimizer = Adam

parser = get_sac_parser()
parser.add_argument('--env_name', default="MountainCarContinuous-v0")

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

logdir = f"./tensorboard_log/{args.env_name}/"
# logdir += "dummy"
modeldir = f"./model/{args.env_name}/"
bufferdir = f"./buffer/{args.env_name}/"

logfile = get_logfile_name(args)

os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(log_dir=logdir + logfile)

squasher = TanhSquasher()

policy = GaussianPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
                        log_std_min=args.log_std_min, log_std_max=args.log_std_max)

sac = SAC(action_dim=action_dim, state_dim=state_dim, hidden_dim=hidden_dim,
          discount_factor=args.discount_factor, optimizer=optimizer, policy_lr=args.policy_lr, critic_lr=args.critic_lr,
          value_lr=args.value_lr, writer=writer, scale_reward=args.scale_reward, reparam=args.reparam,
          target_update_interval=args.target_update_interval, soft_update_tau=args.soft_update_tau,
          td3_update_interval=args.td3_update_interval, policy_weight_decay=args.policy_weight_decay,
          critic_weight_decay=args.critic_weight_decay, value_weight_decay=args.value_weight_decay,
          grad_clip=args.grad_clip, loss_clip=args.loss_clip, clip_val_grad=args.clip_val_grad,
          deterministic=args.deterministic, clip_val_loss=args.clip_val_loss, policy=policy)

buffer = Buffer(capacity=args.buffer_capacity)
rewards = []
update_count = 1
max_reward = -np.inf


def ld_to_dl(batch_list_of_dicts):
    batch_dict_of_lists = {}
    for k in batch_list_of_dicts[0].keys():
        batch_dict_of_lists[k] = []
    for dictionary in batch_list_of_dicts:
        for k in batch_list_of_dicts[0].keys():
            batch_dict_of_lists[k].append(dictionary[k])
    return batch_dict_of_lists


for cur_episode in range(args.num_episodes):
    print(f"Starting episode {cur_episode}")

    state = torch.Tensor(env.reset())
    done = False
    timestep = 0
    episode_reward = 0

    while not done and timestep <= args.max_time_steps:
        action = sac.policy.get_action(state, squasher=squasher, deterministic=False, evaluate=False).detach()
        observation, reward, done, _ = gym_torchify(env.step(action))
        sample = dict(state=state, action=action, reward=reward, next_state=observation, done=done)
        buffer.add(sample)
        if len(buffer) > args.sample_batch_size:
            for num_update in range(args.updates_per_step):
                update_count += 1
                batch_list_of_dicts = buffer.sample(batch_size=args.sample_batch_size)
                batch_dict_of_lists = ld_to_dl(batch_list_of_dicts)

                """ Combined Experience replay. Add online transition too.
                """
                for k in batch_list_of_dicts[0].keys():
                    batch_dict_of_lists[k].append(sample[k])
                sac.policy_update(batch_dict_of_lists, update_number=update_count, squasher=squasher,
                                  reparam=sac.reparam, evaluate=True)

        episode_reward += reward
        state = observation
        timestep += 1

    if episode_reward > max_reward:
        max_reward = episode_reward
        # save current best model
        print(f"\nNew best model with reward {max_reward}")
        sac.save_model(all_nets_path=modeldir + logfile + "/", env_name=args.env_name, info='best')

    if cur_episode % args.save_iter == 0:
        print(f"\nSaving periodically - iteration {cur_episode}")
        sac.save_model(all_nets_path=modeldir + logfile + "/", env_name=args.env_name, info="periodic")
        buffer.save_buffer(path=bufferdir + logfile + "/", info=args.env_name)

    sac.writer.add_scalar("Episode Reward", episode_reward, cur_episode)
    rewards.append(episode_reward)

    sac.writer.add_scalar("Episode Length", timestep, cur_episode)
    # for name, param in sac.policy.named_parameters():
    #     sac.writer.add_histogram("policy_" + name, param, cur_episode)
    # for name, param in sac.value.named_parameters():
    #     sac.writer.add_histogram("value_" + name, param, cur_episode)
    # for name, param in sac.critics[0].named_parameters():
    #     sac.writer.add_histogram("critic1_" + name, param, cur_episode)
    # for name, param in sac.critics[1].named_parameters():
    #     sac.writer.add_histogram("critic2_" + name, param, cur_episode)
