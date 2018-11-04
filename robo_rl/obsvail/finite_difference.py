import copy
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchfunc
from osim.env import ProstheticsEnv
from pros_ai import get_policy_observation
from robo_rl.common import LinearNetwork, xavier_initialisation, None_grad
from tensorboardX import SummaryWriter
from torch.optim import Adam, SGD
import torch.autograd as autograd

torch.manual_seed(0)

env = ProstheticsEnv(visualize=False)

logdir = "./finite_difference/log/"
modeldir = "./finite_difference/model/"

lr = 0.01
num_iterations = 500
epsilon = 0.01
hidden_dim = [512]

logfile = "_reward_SGD"
logfile += f"lr_{lr}_num_iterations_{num_iterations}_hidden_dim={hidden_dim}_epsilon={epsilon}"
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
    LinearNetwork(layers_size=layers_size, final_layer_function=torch.sigmoid, activation_function=torchfunc.elu,
                  is_layer_norm=False) for _ in range(trajectory_length)])

policy.apply(xavier_initialisation)

# policy_optimiser = Adam(policy.parameters(), lr=lr)
policy_optimiser = SGD(policy.parameters(), lr=lr)


def finite_difference(action, index):
    env.reset()
    action_plus = copy.deepcopy(action)
    action_plus[index] += epsilon
    _, reward_plus, _, _ = env.step(action_plus.detach().numpy(), project=False)

    action_minus = copy.deepcopy(action)
    action_minus[index] -= epsilon
    _, reward_minus, _, _ = env.step(action_minus.detach().numpy(), project=False)

    return torch.Tensor([(reward_plus - reward_minus) / (2 * epsilon)]).mean()


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
        actions_grads = []
        actions = []
        action_out_of_bound_penalty = 0
        while not done and timestep <= max_timestep:
            action = policy[timestep].forward(observation[:, 0])
            print(action)
            action = torch.clamp(action, min=0, max=1)
            actions.append(action)

            action_grads = []
            for action_index in range(action_dim):
                action_grads.append(finite_difference(action.detach(), action_index))
            actions_grads.append(action_grads)

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

        print(actions[0], actions_grads[0])

        None_grad(policy_optimiser)
        autograd.backward(tensors=actions[0],
                          grad_tensors=torch.Tensor(actions_grads[0]), retain_graph=True)

        policy_optimiser.step()

    print("Saving model")
    model_path = modeldir + logfile + "best_model.pt"
    torch.save(policy.state_dict(), model_path)
