import os

import robo_rl.common.utils.nn_utils as nn_utils
import robo_rl.common.utils.utils as utils
import torch
import torch.nn as nn
from robo_rl.common import LinearQNetwork, LinearValueNetwork
from robo_rl.common.utils import soft_update, hard_update
from robo_rl.sac import GaussianPolicy


def n_critics(state_dim, action_dim, hidden_dim, num_q):
    q_networks = nn.ModuleList([LinearQNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
                                for _ in range(num_q)])

    return q_networks


class SAC:
    def __init__(self, action_dim, state_dim, hidden_dim, writer, squasher, optimizer, discount_factor=0.99,
                 scale_reward=3, policy_lr=0.0003, critic_lr=0.0003, value_lr=0.0003,
                 reparam=True, target_update_interval=1, soft_update_tau=0.005,
                 td3_update_interval=100, deterministic=False, weight_decay=0.001,
                 grad_clip=False, loss_clip=False, clip_val_grad=0.01, clip_val_loss=100,
                 log_std_min=-20, log_std_max=-2):
        self.writer = writer
        self.deterministic = deterministic
        self.squasher = squasher
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.discount_factor = discount_factor
        self.scale_reward = scale_reward
        self.reparam = reparam
        self.target_update_interval = target_update_interval
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.critic_lr = critic_lr
        self.soft_update_tau = soft_update_tau
        self.td3_update_interval = td3_update_interval
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.loss_clip = loss_clip
        self.clip_val_grad = clip_val_grad
        self.clip_val_loss = clip_val_loss
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.value = LinearValueNetwork(state_dim=self.state_dim, hidden_dim=self.hidden_dim)
        self.value_target = LinearValueNetwork(state_dim=self.state_dim, hidden_dim=self.hidden_dim)
        self.policy = GaussianPolicy(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim,
                                     log_std_min=self.log_std_min, log_std_max=self.log_std_max)
        self.critics = n_critics(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim,
                                 num_q=2)
        self.value.apply(nn_utils.xavier_initialisation)
        self.critics.apply(nn_utils.xavier_initialisation)
        self.policy.apply(nn_utils.xavier_initialisation)

        self.value_optimizer = optimizer(self.value.parameters(), lr=self.value_lr, weight_decay=self.weight_decay)
        self.policy_optimizer = optimizer(self.policy.parameters(), lr=self.policy_lr, weight_decay=self.weight_decay)
        self.critic1_optimizer = optimizer(self.critics[0].parameters(), lr=self.critic_lr,
                                           weight_decay=self.weight_decay)
        self.critic2_optimizer = optimizer(self.critics[1].parameters(), lr=self.critic_lr,
                                           weight_decay=self.weight_decay)

        hard_update(target=self.value_target, original=self.value)

    def policy_update(self, batch, update_number):
        mse_loss = nn.MSELoss()
        # target_clip_min = -self.scale_reward/(1-self.discount_factor)
        # target_clip_max = 0

        """batch is a dict from replay buffer"""
        state_batch = torch.stack(batch['state']).detach()
        action_batch = torch.stack(batch['action']).detach()
        reward_batch = torch.stack(batch['reward']).detach()
        next_state_batch = torch.stack(batch['next_state']).detach()
        done_batch = torch.stack(batch['done']).detach()

        value = self.value(state_batch)
        target_value = self.value_target(next_state_batch)

        # Q^ = scaled reward + discount_factor * exp_target_value(st+1)
        q_hat_buffer = self.scale_reward * reward_batch + (1 - done_batch) * self.discount_factor * target_value
        # q_hat_buffer = torch.clamp(q_hat_buffer, min=target_clip_min, max=target_clip_max)

        # Q values for state and action taken from given batch (sampled from replay buffer)
        q1_buffer = self.critics[0](torch.cat([state_batch, action_batch], 1))
        q2_buffer = self.critics[1](torch.cat([state_batch, action_batch], 1))

        # JQ  ----------------
        q1_val_loss = 0.5 * mse_loss(q1_buffer, q_hat_buffer.detach())
        q2_val_loss = 0.5 * mse_loss(q2_buffer, q_hat_buffer.detach())

        policy_action, log_prob = self.policy.get_action(state_batch, squasher=self.squasher, reparam=self.reparam,
                                                         evaluate=True)

        # to calculate JV and Jpi state is sampled from buffer but action is sampled from policy
        # Min of 2 q value is used in eqn(6)
        q1_current_policy = self.critics[0](torch.cat([state_batch, policy_action], 1))
        q2_current_policy = self.critics[1](torch.cat([state_batch, policy_action], 1))

        min_q_value = torch.min(q1_current_policy, q2_current_policy)

        """ JV= Est~D[ 0.5(V(st)- (Eat~pi [(Qmin (st,at) - logpi]) )^2]
        actions sampled according to current policy and not replay buffer
        v_target = Eat~pi (Qmin (st,at) - logpi
        """
        v_target = min_q_value - log_prob
        # v_target = torch.clamp(v_target, min=target_clip_min, max=target_clip_max)
        value_loss = 0.5 * mse_loss(value, v_target.detach())

        # policy loss
        if self.reparam:
            # reparameterization trick.
            # zero grad on critic will clear there policy loss grads
            policy_loss = (log_prob - min_q_value).mean()
        else:
            policy_loss = (log_prob * (log_prob - min_q_value + value.detach())).mean()
        action_penalty = (policy_action ** 2).mean()
        # policy_loss += action_penalty

        self.critic1_optimizer.zero_grad()
        if self.loss_clip:
            q1_val_loss = torch.clamp(q1_val_loss, min=-self.clip_val_loss, max=self.clip_val_loss)
        q1_val_loss.backward()
        if self.grad_clip:
            for k, v in self.critics[0].named_parameters():
                v.grad = torch.clamp(v.grad, min=-self.clip_val_grad, max=self.clip_val_grad)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        if self.loss_clip:
            q2_val_loss = torch.clamp(q2_val_loss, min=-self.clip_val_loss, max=self.clip_val_loss)
        q2_val_loss.backward()
        if self.grad_clip:
            for k, v in self.critics[1].named_parameters():
                v.grad = torch.clamp(v.grad, min=-self.clip_val_grad, max=self.clip_val_grad)
        self.critic2_optimizer.step()

        self.value_optimizer.zero_grad()
        if self.loss_clip:
            value_loss = torch.clamp(value_loss, min=-self.clip_val_loss, max=self.clip_val_loss)
        value_loss.backward()
        if self.grad_clip:
            for k, v in self.value.named_parameters():
                v.grad = torch.clamp(v.grad, min=-self.clip_val_grad, max=self.clip_val_grad)
        self.value_optimizer.step()

        if update_number % self.td3_update_interval == 0:
            self.policy_optimizer.zero_grad()
            if self.loss_clip:
                policy_loss = torch.clamp(policy_loss, min=-self.clip_val_loss, max=self.clip_val_loss)
            policy_loss.backward()
            if self.grad_clip:
                for k, v in self.policy.named_parameters():
                    v.grad = torch.clamp(v.grad, min=-self.clip_val_grad, max=self.clip_val_grad)
            self.policy_optimizer.step()

        if self.target_update_interval > 1:
            if update_number % self.target_update_interval == 0 and self.deterministic is False:
                hard_update(original=self.value, target=self.value_target)
        else:
            if self.deterministic is False:
                soft_update(original=self.value, target=self.value_target, t=self.soft_update_tau)

        self.writer.add_scalar("Value mean", value.mean(), global_step=update_number)
        self.writer.add_scalar("Value target next state mean", target_value.mean(), global_step=update_number)
        self.writer.add_scalar("Q hat mean", q_hat_buffer.mean(), global_step=update_number)
        self.writer.add_scalar("Q1 buffer mean", q1_buffer.mean(), global_step=update_number)
        self.writer.add_scalar("Q2 buffer mean", q2_buffer.mean(), global_step=update_number)
        self.writer.add_scalar("Log prob mean", log_prob.mean(), global_step=update_number)
        self.writer.add_scalar("Q1 current mean", q1_current_policy.mean(), global_step=update_number)
        self.writer.add_scalar("Q2 current mean", q2_current_policy.mean(), global_step=update_number)
        self.writer.add_scalar("Min Q current mean", min_q_value.mean(), global_step=update_number)
        self.writer.add_scalar("Target value for loss mean", v_target.mean(), global_step=update_number)
        self.writer.add_scalar("action penalty mean", action_penalty.mean(), global_step=update_number)
        self.writer.add_scalar("action mean", policy_action.mean(), global_step=update_number)

        self.writer.add_scalar("Value loss", value_loss, global_step=update_number)
        self.writer.add_scalar("Q Value 1 loss", q1_val_loss, global_step=update_number)
        self.writer.add_scalar("Q Value 2 loss", q2_val_loss, global_step=update_number)
        self.writer.add_scalar("Policy loss", policy_loss, global_step=update_number)

    def save_model(self, env_name, all_nets_path=None, actor_path=None, critic_path=None, value_path=None, info=1):
        if all_nets_path is not None:
            actor_path = all_nets_path
            value_path = all_nets_path
            critic_path = all_nets_path
        if actor_path is None:
            actor_path = f'model/{env_name}/'
        os.makedirs(actor_path, exist_ok=True)
        if critic_path is None:
            critic_path = f'model/{env_name}/'
        os.makedirs(critic_path, exist_ok=True)
        if value_path is None:
            value_path = f'model/{env_name}/'
        os.makedirs(value_path, exist_ok=True)

        utils.print_heading("Saving actor,critic,value network parameters")
        torch.save(self.policy.state_dict(), actor_path + f"actor_{info}.pt")
        torch.save(self.value.state_dict(), value_path + f"value_{info}.pt")
        torch.save(self.critics.state_dict(), critic_path + f"critics_{info}.pt")
        utils.heading_decorator(bottom=True, print_req=True)

    def load_model(self, actor_path=None, critic_path=None, value_path=None):
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

    def get_action(self, state, deterministic=False, evaluate=False):
        return self.policy.get_action(state, self.squasher, deterministic=deterministic, evaluate=evaluate)
