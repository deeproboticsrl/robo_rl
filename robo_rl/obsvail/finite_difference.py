import copy
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchfunc
from osim.env import ProstheticsEnv
from pros_ai import get_policy_observation
from robo_rl.common import LinearNetwork, no_activation, xavier_initialisation, None_grad
from tensorboardX import SummaryWriter
from torch.optim import Adam

torch.manual_seed(0)

env = ProstheticsEnv(visualize=False)

logdir = "./finite_difference/log/"
modeldir = "./finite_difference/model/"

lr = 0.0001
num_iterations = 500
epsilon = 0.0001
hidden_dim = [512]

logfile = f"lr_{lr}_num_iterations_{num_iterations}_hidden_dim={hidden_dim}_epsilon={epsilon}"
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
    LinearNetwork(layers_size=layers_size, final_layer_function=torch.sigmoid, activation_function=torchfunc.elu
                  ) for _ in range(trajectory_length)])

policy.apply(xavier_initialisation)

policy_optimiser = Adam(policy.parameters(), lr=lr)


def finite_difference(action, index):
    env.reset()
    action_plus = copy.deepcopy(action)
    action_plus[index] += epsilon
    observation_plus, _, _, _ = env.step(action_plus.detach().numpy(), project=False)
    observation_plus = get_policy_observation(observation_plus)

    action_minus = copy.deepcopy(action)
    action_minus[index] -= epsilon
    observation_minus, _, _, _ = env.step(action_minus.detach().numpy(), project=False)
    observation_minus = get_policy_observation(observation_minus)

    return torch.Tensor((observation_plus - observation_minus) / (2 * epsilon))


# minimise loss with experts incrementally
for max_timestep in range(1):

    print(f"Starting for timestep {max_timestep}")

    for iteration in range(num_iterations):
        print(max_timestep * num_iterations + iteration)

        policy_trajectory = []
        observation = torch.Tensor(get_policy_observation(env.reset(project=False)))
        done = False
        timestep = 0
        episode_reward = 0

        loss_grad = []
        action_grads = []
        actions = []
        while not done and timestep <= max_timestep:
            action = policy[timestep].forward(observation[:, 0])
            action = torch.clamp(action,min=0,max=1)
            actions.append(action)

            for action_index in range(action_dim):
                action_grads.append(finite_difference(action.detach(), action_index))

            observation, reward, done, _ = env.step(action.detach().numpy(), project=False)
            observation = get_policy_observation(observation)

            expert_loss_grad = []
            expert_loss = []
            for expert_trajectory in expert_trajectories:
                expert_loss_grad.append(sum(2 * (observation - expert_trajectory[0]["state"])) / state_dim)
                expert_loss.append(sum((observation - expert_trajectory[0]["state"]) ** 2) / state_dim)
            loss_grad.append(sum(expert_loss_grad) / num_experts)

            writer.add_histogram(f"Imitation loss {timestep}", np.array(expert_loss),
                                 global_step=(max_timestep - timestep) * num_iterations + iteration)
            writer.add_scalar(f"Imitation loss mean {timestep}", sum(expert_loss) / num_experts,
                              global_step=(max_timestep - timestep) * num_iterations + iteration)

            observation = torch.Tensor(observation)
            episode_reward += reward
            timestep += 1

        writer.add_scalar("Episode reward", episode_reward, global_step=max_timestep * num_iterations + iteration)

        policy_loss = 0
        for i in range(len(loss_grad)):
            print(loss_grad[i],action_grads[i],actions[i])
            policy_loss += (torch.Tensor(loss_grad[i]) * actions[i] * action_grads[i]).sum()
            # policy_loss += torch.Tensor([out_of_bound_penalty_episode[i]])

        None_grad(policy_optimiser)
        policy_loss.backward()
        policy_optimiser.step()

    print("Saving model")
    model_path = modeldir + logfile + "best_model.pt"
    torch.save(policy.state_dict(), model_path)
