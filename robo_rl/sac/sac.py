import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from robo_rl.sac.tanh_gaussian_policy import TanhGaussianPolicy
from robo_rl.common.networks.value_network import LinearQNetwork, LinearValueNetwork

# ASSUMPTION : non deterministic
def soft_update(original, target, t):
    # zip(a,b) is same as [a.b]
    for original_param, target_param in zip(original.parameters(), target.parameters()):
        target_param.data.copy_(original_param.data * t + target_param * (1 - t))
        ## check copy_ parameter : Something on cpu or gpu, also no need for return as it changes in self


def hard_update(original, target):
    for original_param, target_param in zip(original.parameters(), target.parameters()):
        target_param.data.copy_(original_param.data)


class SAC:
    def __init__(self, action_dim, state_dim, **args):  ## TO do
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = args['gamma']
        self.scale_reward = args['scale_reward']
        self.reparam = args['reparam']
        self.deterministic = args['deterministic']
        self.target_update_interval = args['target_update_interval']
        self.hidden_dim = args['hidden_dim']
        self.lr = args['lr']

        ## hard and soft updates

        self.value = LinearValueNetwork(self.state_dim, [self.hidden_dim, self.hidden_dim])
        self.value_target = LinearValueNetwork(self.state_dim, [self.hidden_dim, self.hidden_dim])
        self.value_optimizer = Adam(self.value.parameters(), lr=self.lr)
        hard_update(self.value, self.value_target)

        self.policy = TanhGaussianPolicy(self.state_dim, self.action_dim, [self.hidden_dim, self.hidden_dim])
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.lr)

        self.critic = LinearQNetwork(self.state_dim, self.action_dim, [self.hidden_dim, self.hidden_dim])
        self.critic_optimizer = Adam(self.policy.parameters(), lr=self.lr)

    ## batch is dict from replay buffer
    def policy_update(self, batch):
        state_batch = torch.Tensor(batch['state'])
        action_batch = torch.Tensor(batch['action'])
        reward_batch = torch.Tensor(batch['reward'])
        next_state_batch = batch['next_state']
        done_batch = batch['done']
        #not_done_batch = np.logical_not(done_batch)  ## batch_size * 1
        exp_q1 = self.critic(state_batch, action_batch)
        exp_q2 = self.critic(state_batch,action_batch)
        new_action, z, log_prob, mean, log_std = self.policy.evaluation(state_batch,reparametrize=self.reparam)


        # now to stabilize training for soft value
        # actions sampled according to current policy and not replay buffer
        # use minimum of 2 qa values for value grad and policy grad
        # reshape reward and done batch
        # assuming stochaistic
        if self.deterministic == False:
            exp_value = self.value(state_batch)
            exp_target_value = self.value_target(next_state_batch)
            ##Q^ = scaled reward + gamma * exp_target_value(st+1)
            next_q_value = self.scale_reward * reward_batch +  (1-done_batch) * self.gamma * exp_target_value


        # JQ
        q1_val_loss= nn.MSELoss(exp_q1,next_q_value)
        q2_val_loss=nn.MSELoss(exp_q2,next_q_value)

        ## to calculate JV state is sampled from buffer but action is sampled from policy
        # Min of 2 q value is used in eqn(6)
        q1_new=self.critic(state_batch,new_action)
        q2_new = self.critic(state_batch,new_action)
        expected_new_q_value = torch.min(exp_q1,exp_q2)


        ## JV= Est~D[0.5(V(st)- (Eat~pi (Qmin (st,at) - logpi )^2
        next_value = expected_new_q_value-log_prob
        value_loss = nn.MSELoss(exp_value-next_value)

        # policy loss
        if self.reparam=True:
            # reparameterization trick
            policy_loss = (log_prob-expected_new_q_value).mean()

        self.critic_optimizer.zero_grad()
        q1_val_loss.backward()
        self.critic_optimizer.step()

        self.critic_optimizer.zero_grad()
        q2_val_loss.backward()
        self.critic_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.value_optimizer.step()

        if update_number % self.target_update_interval==0  and self.deterministic==False:
            soft_update(self.value,self.value_target,self.t)
            





