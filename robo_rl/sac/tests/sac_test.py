import os

import gym
import numpy as np
import torch
import torch.nn as nn
from robo_rl.common.utils.nn_utils import print_network_architecture
from robo_rl.common.utils.utils import print_heading
from robo_rl.sac.softactorcritic import SAC
from robo_rl.sac.squasher import TanhSquasher
from tensorboardX import SummaryWriter

env = gym.make("FetchReach-v1")

# Set seeds everywhere
seed = 0
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.spaces["observation"].shape[0]

hidden_dim = [256, 256]

squasher = TanhSquasher()

logdir = "./tensorboard_log/"
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(log_dir=logdir)

sac = SAC(state_dim=state_dim, action_dim=action_dim, writer=writer, hidden_dim=hidden_dim)

print_heading("Architecture of value network")
print_network_architecture(sac.value)

print_heading("Architecture of Q-value networks (critics)")
print_network_architecture(sac.critics)

print_heading("Architecture of policy")
print_network_architecture(sac.policy)

print_heading("Check initialisation of networks using random observation and action")
state = torch.Tensor(env.reset()["observation"])
action = sac.policy.get_action(state, squasher=squasher, evaluate=False)
state_action = torch.cat([state, action], 0)
print("Value ".ljust(20), sac.value(state))
print("Target Value ".ljust(20), sac.value_target(state))
print("Critic 1 : Q Value".ljust(20), sac.critics[0](state_action))
print("Critic 2 : Q Value".ljust(20), sac.critics[1](state_action))
print("Policy ".ljust(20), action)

state_batch = [state]
action_batch = [action]
reward_batch = []
next_state_batch = []
done_batch = []
num_steps = 2
for i in range(num_steps):
    next_state, reward, done, info = env.step(action.detach().numpy())
    next_state_batch.append(torch.Tensor(next_state["observation"]))
    reward_batch.append(torch.Tensor([reward]))
    # done will be False since just reset environment
    done_batch.append(torch.Tensor([done]))

    if i < num_steps - 1:
        state_batch.append(next_state_batch[i])
        action = sac.policy.get_action(next_state_batch[i], squasher=squasher, evaluate=False)
        action_batch.append(action)

state_batch = torch.stack(state_batch).detach()
action_batch = torch.stack(action_batch).detach()
reward_batch = torch.stack(reward_batch).detach()
next_state_batch = torch.stack(next_state_batch).detach()
done_batch = torch.stack(done_batch).detach()

print_heading("Calculations for JQ")
q_hat_not_done = sac.scale_reward * reward_batch + \
                 sac.discount_factor * (1 - done_batch) * sac.value_target(next_state_batch)
q_hat_done = sac.scale_reward * reward_batch + \
             sac.discount_factor * done_batch * sac.value_target(next_state_batch)
q_1 = sac.critics[0](torch.cat([state_batch, action_batch], 1))
q_2 = sac.critics[1](torch.cat([state_batch, action_batch], 1))
mse_loss = nn.MSELoss()
q1_loss = mse_loss(q_1, q_hat_not_done.detach())
q2_loss = mse_loss(q_2, q_hat_not_done.detach())

print("Reward".ljust(25), reward_batch[0], reward_batch[1])
print("Scale Factor".ljust(25), sac.scale_reward)
print("q_hat - not done".ljust(25), q_hat_not_done[0], q_hat_not_done[1])
print("q_hat - done".ljust(25), q_hat_done[0], q_hat_done[1])
print("q1 ".ljust(25), q_1[0], q_1[1])
print("q2 ".ljust(25), q_2[0], q_2[1])
print("q1 loss".ljust(25), q1_loss)
print("q2 loss".ljust(25), q2_loss)

print_heading("Update Q1 and Q2")

sac.critic1_optimizer.zero_grad()
q1_loss.backward()
sac.critic1_optimizer.step()

q_1 = sac.critics[0](torch.cat([state_batch, action_batch], 1))
q_2 = sac.critics[1](torch.cat([state_batch, action_batch], 1))
print("Q1 optimised, hence only Q1 should change")
print("q1 ".ljust(25), q_1[0], q_1[1])
print("q2 ".ljust(25), q_2[0], q_2[1])

sac.critic2_optimizer.zero_grad()
q2_loss.backward()
sac.critic2_optimizer.step()
sac.critic2_optimizer.zero_grad()
sac.critic2_optimizer.step()


q_1 = sac.critics[0](torch.cat([state_batch, action_batch], 1))
q_2 = sac.critics[1](torch.cat([state_batch, action_batch], 1))
print("Q2 optimised, hence only Q2 should change")
print("q1 ".ljust(25), q_1[0], q_1[1])
print("q2 ".ljust(25), q_2[0], q_2[1])

