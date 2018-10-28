import os
import pickle

import numpy as np
import torch
from osim.env import ProstheticsEnv
from osim.http.client import Client
from pros_ai import get_policy_observation
from robo_rl.obsvail import get_obsvail_parser, get_logfile_name
from robo_rl.sac import SAC, SigmoidSquasher
from tensorboardX import SummaryWriter
from torch.optim import Adam

optimizer = Adam

parser = get_obsvail_parser()
args = parser.parse_args()
env = ProstheticsEnv(visualize=True)

seed = 0
# seeding
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

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

play_actor = False
submit = False

current_observation = get_policy_observation(env.reset(project=False))

if play_actor:
    actor_path = modeldir + logfile + "/actor_best.pt"

    sac.load_model(actor_path=actor_path)

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

        current_observation = observation
        state = torch.Tensor(np.append(np.append(current_observation, context), 0))
        episode_reward += reward
        timestep += 1
        print(episode_reward, timestep)

    print(episode_reward, timestep)

else:
    if submit:
        remote_base = "http://grader.crowdai.org:1730"
        crowdai_token = "f5969a7bb0466e0da072c72d6eb6d667"

        client = Client(remote_base)

        with open(modeldir + logfile + "/best_trajectory.pkl", "rb") as f:
            trajectory = pickle.load(f)["trajectory"]

        done = False
        timestep = 0

        observation = client.env_create(crowdai_token, env_id='ProstheticsEnv')
        episode_reward = 0

        while True:
            action = trajectory[timestep % 100]["action"]
            [observation, reward, done, info] = client.env_step(action.detach().numpy().tolist(), True)
            episode_reward += reward
            timestep += 1
            print(episode_reward, timestep)
            if done:
                observation = client.env_reset()
                print("Reset")
                if not observation:
                    break

        client.submit()

    else:
        with open(modeldir + logfile + "/best_trajectory.pkl", "rb") as f:
            trajectory = pickle.load(f)["trajectory"]
        done = False
        timestep = 0

        episode_reward = 0
        while not done:
            action = trajectory[timestep % 100]["action"]
            observation, reward, done, _ = env.step(np.array(action), project=False)

            episode_reward += reward
            timestep += 1
            print(episode_reward, timestep)

        print(episode_reward, timestep)
