import os

import numpy as np
import torch
from osim.env import ProstheticsEnv
from pros_ai import get_policy_observation
from robo_rl.obsvail import get_obsvail_parser, get_logfile_name
from robo_rl.sac import SAC, SigmoidSquasher
from tensorboardX import SummaryWriter
from torch.optim import Adam

optimizer = Adam

parser = get_obsvail_parser()
args = parser.parse_args()
env = ProstheticsEnv(visualize=True)

# seeding
env.seed(args.env_seed)
torch.manual_seed(args.env_seed)
np.random.seed(args.env_seed)

observation = env.reset(project=False)
action_dim = env.action_space.shape[0]

"""This is done to allow having different observations for policy and discriminator
"""
policy_state_dim = get_policy_observation(observation).shape[0]

context_dim = 2

# According to VAIL
sac_hidden_dim = [1024, 512]

logdir = "./tensorboard_log/"
# logdir += "dummy"
modeldir = f"./model/ProstheticsEnv/"
bufferdir = f"./buffer/ProstheticsEnv/"
attributesdir = f"./attributes/ProstheticsEnv/"

logfile = get_logfile_name(args)

os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(log_dir=logdir + logfile)

squasher = SigmoidSquasher()

sac = SAC(action_dim=action_dim, state_dim=policy_state_dim + context_dim + 1, hidden_dim=sac_hidden_dim,
          discount_factor=args.discount_factor, optimizer=optimizer, policy_lr=args.policy_lr, critic_lr=args.critic_lr,
          value_lr=args.value_lr, writer=writer, scale_reward=args.scale_reward, reparam=args.reparam,
          target_update_interval=args.target_update_interval, soft_update_tau=args.soft_update_tau,
          td3_update_interval=args.td3_update_interval, squasher=squasher, policy_weight_decay=args.policy_weight_decay,
          critic_weight_decay=args.critic_weight_decay, value_weight_decay=args.value_weight_decay,
          grad_clip=args.grad_clip, loss_clip=args.loss_clip, clip_val_grad=args.clip_val_grad,
          deterministic=args.deterministic, clip_val_loss=args.clip_val_loss, log_std_min=args.log_std_min,
          log_std_max=args.log_std_max)

actor_path = modeldir + logfile + "/actor_best.pt"

sac.load_model(actor_path=actor_path)

current_observation = get_policy_observation(env.reset(project=False))

# Sample random context for the trajectory
context = [np.random.randint(0, 1) for _ in range(context_dim)]
# indicator for absorbing state
state = torch.Tensor(np.append(np.append(current_observation, context), 0))
done = False
timestep = 0

# Episode reward is used only as a metric for performance
episode_reward = 0
while not done:
    action = sac.get_action(state).detach()
    observation, reward, done, _ = env.step(np.array(action), project=False)
    observation = get_policy_observation(observation)
    sample = dict(state=current_observation, action=action, reward=reward, is_absorbing=False,
                  next_state=observation, done=done)

    current_observation = observation
    state = torch.Tensor(np.append(np.append(current_observation, context), 0))
    episode_reward += reward
    timestep += 1
    print(episode_reward, timestep)

print(episode_reward, timestep)
