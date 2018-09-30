import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal

## GMM policy L to be implemented


class TanhGaussianPolicy(nn.Module):

    def __init__(self, state_dim , action_dim, hidden_dim):
        super(TanhGaussianPolicy,self).__init__()

        self.layer1 = nn.Linear (state_dim,hidden_dim)
        self.layer2 = nn.Linear (hidden_dim,hidden_dim)

        self.mean_layer = nn.Linear (hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim,action_dim)


    def forward(self,state):
        y= F.relu(self.layer1(state))
        y= F.relu(self.layer2(y))

        mean = self.mean_layer(y)
        log_std = self.log_std_layer(y)
        return mean ,log_std

    # def action_to_apply(self,state):

    def get_action (self,state, epsilon=1e-6, reparam=True):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        if reparam==True:
            z = normal.rsample()    # reparameterization trick
        else:
            z = normal.sample()

        action = torch.tanh(z)
        # return action
        # log_prob_action =  normal.log_prob(z)-torch.log(1-action.pow(2) + epsilon)
        # log_prob = log_prob.sum(-1, keepdim=True)  # if tensor is 4*4 row sum wil be 4*1
        return action

        # Action bound  squash correction





