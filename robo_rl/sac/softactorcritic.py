import torch
import torch.nn as nn
from torch.optim import Adam
from robo_rl.sac.gaussian_policy import GaussianPolicy
from robo_rl.common.networks.value_network import LinearQNetwork, LinearValueNetwork
import robo_rl.common.utils.nn_utils as nn_utils
import robo_rl.common.utils.utils as utils
import os


# ASSUMPTION : non deterministic
# TODO dfjbnj

def soft_update(original, target, t=1e-2):
    # zip(a,b) is same as [a.b]
    for original_param, target_param in zip(original.parameters(), target.parameters()):
        target_param.data.copy_(original_param.data * t + target_param * (1 - t))
        ## check copy_ parameter : Something on cpu or gpu, also no need for return as it changes in self


def hard_update(original, target):
    for original_param, target_param in zip(original.parameters(), target.parameters()):
        target_param.data.copy_(original_param.data)


def n_critics(state_dim, action_dim, hidden_array, number_q):
    q_networks = []
    for i in range(number_q):
        q_networks.append(LinearQNetwork(state_dim, action_dim, hidden_array))

    return q_networks


class SAC:
    ## TODO select action
    def __init__(self, action_dim, state_dim, hidden_dim=256, discount_factor=0.05, scale_reward=3,
                 reparam=True, deterministic=False, target_update_interval=300, lr=1e-3, soft_update_tau=1e-2,
                 td3=False):

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.discount_factor = discount_factor
        self.scale_reward = scale_reward
        self.reparam = reparam
        self.deterministic = deterministic
        self.target_update_interval = target_update_interval
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.soft_update_tau = soft_update_tau
        self.td3 = td3

        ## hard and soft updates

        self.value = LinearValueNetwork(self.state_dim, [self.hidden_dim, self.hidden_dim])
        self.value_target = LinearValueNetwork(self.state_dim, [self.hidden_dim, self.hidden_dim])
        self.policy = GaussianPolicy(self.state_dim, self.action_dim, [self.hidden_dim, self.hidden_dim])
        self.critic = n_critics(self.state_dim, self.action_dim, [self.hidden_dim, self.hidden_dim], number_q=2)
        nn_utils.xavier_initialisation(self.value, self.policy, self.critic)
        self.value_optimizer = Adam(self.value.parameters(), lr=self.lr)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.lr)
        self.critic_optimizer = Adam(self.policy.parameters(), lr=self.lr)
        hard_update(self.value, self.value_target)

    ## batch is dict from replay buffer
    def policy_update(self, batch, update_number):
        state_batch = torch.Tensor(batch['state'])
        action_batch = torch.Tensor(batch['action'])
        reward_batch = torch.Tensor(batch['reward'])
        next_state_batch = batch['next_state']
        done_batch = batch['done']
        # not_done_batch = np.logical_not(done_batch)  ## batch_size * 1
        exp_q1, exp_q2 = self.critic(state_batch, action_batch)
        new_action, log_prob = self.policy.get_action(state_batch, reparametrize=self.reparam)

        # now to stabilize training for soft value
        # actions sampled according to current policy and not replay buffer
        # use minimum of 2 qa values for value grad and policy grad
        # reshape reward and done batch
        # assuming stochaistic
        if self.deterministic == False:
            exp_value = self.value(state_batch)
            exp_target_value = self.value_target(next_state_batch)
            ##Q^ = scaled reward + discount_factor * exp_target_value(st+1)
            q_target = self.scale_reward * reward_batch + (1 - done_batch) * self.discount_factor * exp_target_value

        # JQ
        q1_val_loss = nn.MSELoss(exp_q1, (q_target.detach()))
        q2_val_loss = nn.MSELoss(exp_q2, q_target.detach())

        ## to calculate JV and Jpi state is sampled from buffer but action is sampled from policy
        # Min of 2 q value is used in eqn(6)
        q1_new = self.critic(state_batch, new_action)
        q2_new = self.critic(state_batch, new_action)
        expected_new_q_value = torch.min(q1_new, q2_new)

        ## JV= Est~D[0.5(V(st)- (Eat~pi (Qmin (st,at) - logpi )^2
        # v_target = Eat~pi (Qmin (st,at) - logpi
        v_target = expected_new_q_value - log_prob
        value_loss = nn.MSELoss(exp_value - v_target.detach())

        # policy loss
        if self.reparam == True:
            # reparameterization trick
            policy_loss = (log_prob - expected_new_q_value.detach()).mean()

        self.critic_optimizer.zero_grad()
        q1_val_loss.backward()
        self.critic_optimizer.step()

        self.critic_optimizer.zero_grad()
        q2_val_loss.backward()
        self.critic_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        if self.td3 == False:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        if self.target_update_interval > 1:

            if update_number % self.target_update_interval == 0 and self.deterministic == False:
                hard_update(self.value, self.value_target)
                # TODO TD3 update

        else:
            if self.deterministic == False:
                soft_update(self.value, self.value_target, self.soft_update_tau)

    def save_model(self, env_name, actor_path, critic_path, value_path, info=1):

        if actor_path is None:
            actor_path = os.makedirs('model/{}/actor_{}'.format(info, env_name), exist_ok=True)
        if critic_path is None:
            critic_path = os.makedirs('model/{}/critic_{}'.format(info, env_name), exist_ok=True)
        if value_path is None:
            value_path = os.makedirs('model/{}/value_{}'.format(info, env_name), exist_ok=True)

        utils.print_heading("Saving actor,critic,value network parameters")
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.value.state_dict(), value_path)
        torch.save(self.critic.state_dict(), critic_path)
        utils.heading_decorator(bottom=True, print_req=True)

    def load_model(self, model, actor_path, critic_path, value_path):
        utils.print_heading(
            "Loading models from paths: \n actor:{} \n critic:{} \n value:{}".format(actor_path, critic_path,
                                                                                     value_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
        if value_path is not None:
            self.value.load_state_dict(torch.load(value_path))

        utils.print_heading('loading done')
