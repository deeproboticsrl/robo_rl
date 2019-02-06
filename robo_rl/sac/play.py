import os

import gym
import numpy as np
import torch
# from osim.env import ProstheticsEnv
from robo_rl.common.utils import gym_torchify
from robo_rl.sac import SAC, TanhSquasher
from robo_rl.sac import get_sac_parser, get_logfile_name
from tensorboardX import SummaryWriter
from torch.optim import Adam

optimizer = Adam

parser = get_sac_parser()
parser.add_argument('--env_name', default="Humanoid-v2")

args = parser.parse_args()
# if args.env_name == "ProstheticsEnv":
#     env = ProstheticsEnv()
# else:
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

sac = SAC(action_dim=action_dim, state_dim=state_dim, hidden_dim=hidden_dim,
          discount_factor=args.discount_factor, optimizer=optimizer, policy_lr=args.policy_lr, critic_lr=args.critic_lr,
          value_lr=args.value_lr, writer=writer, scale_reward=args.scale_reward, reparam=args.reparam,
          target_update_interval=args.target_update_interval, soft_update_tau=args.soft_update_tau,
          td3_update_interval=args.td3_update_interval, policy_weight_decay=args.policy_weight_decay,
          critic_weight_decay=args.critic_weight_decay, value_weight_decay=args.value_weight_decay,
          grad_clip=args.grad_clip, loss_clip=args.loss_clip, clip_val_grad=args.clip_val_grad,
          deterministic=args.deterministic, clip_val_loss=args.clip_val_loss)

# actor_path = f"model/{args.env_name}/actor_periodic.pt"
# actor_path = modeldir + logfile + "/actor_periodic.pt"
actor_path = modeldir + logfile + "/actor_best.pt"

sac.load_model(actor_path=actor_path)

detertministic_eval = 1

import time

episode_rewards = []
for i in range(20):
    observation = env.reset()
    done = False
    timestep = 0
    episode_reward = 0

    success = False
    while not done and timestep <= args.max_time_steps:
        env.render()
        time.sleep(0.02)
        action = sac.get_action(torch.Tensor(observation), deterministic=detertministic_eval).detach()
        # print(action)
        observation, reward, done, info = gym_torchify(env.step(action))
        timestep += 1
        episode_reward += reward

    print(i, episode_reward, timestep)
    episode_rewards.append(episode_reward)

print(f"Average reward {sum(episode_rewards)/len(episode_rewards)}")
print(f"Max reward {max(episode_rewards)}")
print(f"Min reward {min(episode_rewards)}")
