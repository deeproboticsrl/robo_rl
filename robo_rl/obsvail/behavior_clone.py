import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchfunc
from osim.env import ProstheticsEnv
from pros_ai import get_policy_observation
from robo_rl.common import LinearGaussianNetwork, no_activation, xavier_initialisation, None_grad
from tensorboardX import SummaryWriter
from torch.distributions import Normal
from torch.optim import Adam

torch.manual_seed(0)

env = ProstheticsEnv(visualize=False)

logdir = "./behavior_clone/log/"
modeldir = "./behavior_clone/model/"

lr = 10
num_iterations = 500
hidden_dim = [512]

logfile = f"lr_{lr}_num_iterations_{num_iterations}_hidden_dim={hidden_dim}"
logfile += "/"

os.makedirs(modeldir + logfile, exist_ok=True)

writer = SummaryWriter(log_dir=logdir + logfile)

expert_file_path = "./experts/sampled_experts.obs"

# Load experts for speed 3 - context 01
with open(expert_file_path, "rb") as expert_file:
    all_expert_trajectories = pickle.load(expert_file)

expert_trajectories = []

for expert_trajectory in all_expert_trajectories:
    if expert_trajectory["context"][0] == 0 and expert_trajectory["context"][1] == 1:
        expert_trajectories.append(expert_trajectory["trajectory"])

num_experts = len(expert_trajectories)
print("Number of experts".ljust(30), num_experts)

trajectory_length = len(expert_trajectories[0])
print("Trajectory length".ljust(30), trajectory_length)

state_dim = expert_trajectories[0][0]["state"].shape[0]
print("State dimension".ljust(30), state_dim)

action_dim = 19
layers_size = [state_dim]
layers_size.extend(hidden_dim)
layers_size.append(action_dim)

# initialise policy as a list of networks
policy = nn.ModuleList([
    LinearGaussianNetwork(layers_size=layers_size, final_layer_function=no_activation, activation_function=torchfunc.elu
                          ) for _ in range(trajectory_length)])

policy.apply(xavier_initialisation)

policy_optimiser = Adam(policy.parameters(), lr=lr)

# minimise loss with experts incrementally
for max_timestep in range(10):

    print(f"Starting for timestep {max_timestep}")

    for iteration in range(num_iterations):
        print(max_timestep*num_iterations + iteration)

        policy_trajectory = []
        observation = torch.Tensor(get_policy_observation(env.reset(project=False)))
        done = False
        timestep = 0
        episode_reward = 0

        loss = []
        log_prob = []
        out_of_bound_penalty_episode = []
        while not done and timestep <= max_timestep:
            out_of_bound_penalty = torch.Tensor([0])
            mean, log_std = policy[timestep].forward(observation[:, 0])

            for q in range(action_dim):
                if mean[q] < 0:
                    out_of_bound_penalty += -mean[q]
                if mean[q] > 1:
                    out_of_bound_penalty += mean[q] - 1
                if log_std[q] > -1:
                    out_of_bound_penalty += log_std[q] + 1
            out_of_bound_penalty_episode.append(out_of_bound_penalty)

            log_std = torch.clamp(log_std, min=-20, max=-1)

            std = torch.exp(log_std)
            action_distribution = Normal(mean, std)
            action = action_distribution.rsample()
            log_prob.append(action_distribution.log_prob(action))

            observation, reward, done, _ = env.step(action.detach().numpy(), project=False)
            observation = get_policy_observation(observation)

            expert_losses = []
            for expert_trajectory in expert_trajectories:
                expert_losses.append(sum((observation - expert_trajectory[0]["state"]) ** 2) / state_dim)
            loss.append(sum(expert_losses) / num_experts)

            writer.add_histogram(f"Imitation loss {timestep}", np.array(expert_losses),
                                 global_step=(max_timestep - timestep) * num_iterations + iteration)
            writer.add_scalar(f"Imitation loss mean {timestep}", sum(expert_losses) / num_experts,
                              global_step=(max_timestep - timestep) * num_iterations + iteration)
            writer.add_scalar(f"Out of bound penalty {timestep}", out_of_bound_penalty,
                              global_step=(max_timestep - timestep) * num_iterations + iteration)

            observation = torch.Tensor(observation)
            episode_reward += reward
            timestep += 1

        writer.add_scalar("Episode reward", episode_reward, global_step=max_timestep * num_iterations + iteration)

        policy_loss = 0
        for i in range(len(loss)):
            policy_loss += torch.Tensor(loss[i]) * log_prob[i] + out_of_bound_penalty_episode[i]
            # policy_loss += torch.Tensor([out_of_bound_penalty_episode[i]])

        None_grad(policy_optimiser)
        policy_loss.sum().backward()
        policy_optimiser.step()

    print(loss)
    print("Saving model")
    model_path = modeldir + logfile + "best_model.pt"
    torch.save(policy.state_dict(), model_path)
