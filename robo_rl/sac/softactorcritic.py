import os

import robo_rl.common.utils.nn_utils as nn_utils
import robo_rl.common.utils.utils as utils
import torch
import torch.nn as nn
from robo_rl.common.networks.value_network import LinearQNetwork, LinearValueNetwork
from robo_rl.common.utils.nn_utils import soft_update, hard_update
from robo_rl.sac.gaussian_policy import GaussianPolicy
from torch.optim import Adam



def n_critics(state_dim, action_dim, hidden_dim, num_q):
    q_networks = nn.ModuleList([LinearQNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
                                for _ in range(num_q)])

    return q_networks


class SAC:
    def __init__(self, action_dim, state_dim, hidden_dim, writer, discount_factor=0.99, scale_reward=3,
                 reparameterize=True, target_update_interval=1000, lr=3e-4, soft_update_tau=0.005,
                 td3=False):

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.writer = writer
        self.discount_factor = discount_factor
        self.scale_reward = scale_reward
        self.reparametrize = reparameterize
        self.target_update_interval = target_update_interval
        self.lr = lr
        self.soft_update_tau = soft_update_tau
        self.td3 = td3

        self.value = LinearValueNetwork(state_dim=self.state_dim, hidden_dim=self.hidden_dim)
        self.value_target = LinearValueNetwork(state_dim=self.state_dim, hidden_dim=self.hidden_dim)
        self.policy = GaussianPolicy(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim)
        self.critics = n_critics(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim,
                                 num_q=2)
        nn_utils.xavier_initialisation(self.value, self.policy, self.critics)

        self.value_optimizer = Adam(self.value.parameters(), lr=self.lr)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.lr)
        self.critic1_optimizer = Adam(self.critics[0].parameters(), lr=self.lr)
        self.critic2_optimizer = Adam(self.critics[1].parameters(), lr=self.lr)

        hard_update(target=self.value, original=self.value_target)

    def policy_update(self, batch, update_number):
        mse_loss = nn.MSELoss()

        """batch is a dict from replay buffer"""
        state_batch = torch.Tensor(batch['state']).detach()
        action_batch = torch.Tensor(batch['action']).detach()
        reward_batch = torch.Tensor(batch['reward']).detach()
        next_state_batch = torch.Tensor(batch['next_state']).detach()
        done_batch = torch.Tensor(batch['done']).detach()

        target_value_D = self.value_target(next_state_batch)

        # Q^ = scaled reward + discount_factor * exp_target_value(st+1)
        Q_hat_D = self.scale_reward * reward_batch + (1 - done_batch) * self.discount_factor * target_value_D

        # Q values for state and action taken from given batch (sampled from replay buffer)
        Q1_D = self.critics[0](torch.cat([state_batch, action_batch], 1))
        Q2_D = self.critics[1](torch.cat([state_batch, action_batch], 1))

        # JQ  ----------------
        q1_val_loss = mse_loss(Q1_D, Q_hat_D.detach())
        q2_val_loss = mse_loss(Q2_D, Q_hat_D.detach())

        policy_action, log_prob = self.policy.get_action(state_batch, reparametrize=self.reparametrize)

        # to calculate JV and Jpi state is sampled from buffer but action is sampled from policy
        # Min of 2 q value is used in eqn(6)
        q1_new, q2_new = self.critics(state_batch, policy_action)

        expected_new_q_value = torch.min(q1_new, q2_new)

        """ JV= Est~D[ 0.5(V(st)- (Eat~pi [(Qmin (st,at) - logpi]) )^2]
        actions sampled according to current policy and not replay buffer
        v_target = Eat~pi (Qmin (st,at) - logpi
        """
        v_target = expected_new_q_value - log_prob
        value_loss = mse_loss(value_D - v_target.detach())

        # policy loss
        if self.reparametrize:
            # reparameterization trick
            policy_loss = (log_prob - expected_new_q_value.detach()).mean()
            # TODO : ADD regularization losses


        self.critics_optimizer.zero_grad()
        q1_val_loss.backward()
        self.critics_optimizer.step()

        self.critics_optimizer.zero_grad()
        q2_val_loss.backward()
        self.critics_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        if self.td3 is False:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        if self.target_update_interval > 1:

            if update_number % self.target_update_interval == 0 and self.deterministic is False:
                hard_update(self.value, self.value_target)
                # TODO TD3 update

        else:
            if self.deterministic is False:
                soft_update(self.value, self.value_target, self.soft_update_tau)

    def save_model(self, env_name, actor_path, critic_path, value_path, info=1):

        if actor_path is None:
            actor_path = 'model/{}/actor_{}'.format(info, env_name)
        os.makedirs(actor_path, exist_ok=True)
        if critic_path is None:
            critic_path = 'model/{}/critic_{}'.format(info, env_name)
        os.makedirs(critic_path, exist_ok=True)
        if value_path is None:
            value_path = os.makedirs('model/{}/value_{}'.format(info, env_name), exist_ok=True)

        utils.print_heading("Saving actor,critic,value network parameters")
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.value.state_dict(), value_path)
        torch.save(self.critics.state_dict(), critic_path)
        utils.heading_decorator(bottom=True, print_req=True)

    def load_model(self, model, actor_path, critic_path, value_path):
        utils.print_heading(
            "Loading models from paths: \n actor:{} \n critic:{} \n value:{}".format(actor_path, critic_path,
                                                                                     value_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critics.load_state_dict(torch.load(critic_path))
        if value_path is not None:
            self.value.load_state_dict(torch.load(value_path))

        utils.print_heading('loading done')
