import torch
import torch.nn.functional as torchfunc
from robo_rl.common import LinearGaussianNetwork
from robo_rl.common import no_activation
from torch.distributions import Normal
import numpy as np


def gaaf_relu(x):
    k = 10000
    g = (x*k - torch.floor(x*k) - 0.5)/k
    s = torch.sigmoid(x-0.5)
    y = torchfunc.relu(x) + g * s
    return y


# GMM policy L to be implemented

class GaussianPolicy(LinearGaussianNetwork):

    def __init__(self, state_dim, action_dim, hidden_dim, is_layer_norm=True, log_std_min=-20, log_std_max=2):

        layers_size = [state_dim]
        layers_size.extend(hidden_dim)
        layers_size.append(action_dim)
        super().__init__(layers_size=layers_size, is_layer_norm=is_layer_norm, final_layer_function=no_activation,
                         activation_function=torchfunc.relu)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        """ returns mean and log_std after a forward pass through a linear neural network
        """
        return super().forward(state)

    def get_action(self, state, squasher, epsilon=1e-6, reparam=True, deterministic=False, evaluate=True, info=False):

        mean, log_std = self.forward(state)

        # limit std. neither too stochastic nor too deterministic
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        std = torch.exp(log_std)

        normal = Normal(mean, std)

        if deterministic:
            z = mean
        else:
            if reparam:
                z = normal.rsample()  # reparameterization trick
            else:
                z = normal.sample()

        action = squasher.squash(z)

        if not evaluate:
            return action

        if deterministic:
            """pmf becomes delta function"""
            log_prob = torch.Tensor([np.inf])
        else:
            log_prob = normal.log_prob(z) - torch.log(squasher.derivative(z) + epsilon)

            # If tensor is 5*4, then the row sum wil be 5*1, corresponding to batch size of 5
            log_prob = log_prob.sum(-1, keepdim=True)

        if info:
            return action, log_prob, z, mean, std
        else:
            return action, log_prob

    def compute_log_prob_action(self, state, squasher, action, epsilon=1e-6):

        mean, log_std = self.forward(state)

        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        std = torch.exp(log_std)

        normal = Normal(mean, std)

        z = squasher.inverse(action)
        log_prob = normal.log_prob(z) - torch.log(squasher.derivative(z) + epsilon)

        # If tensor is 5*4, then the row sum wil be 5*1, corresponding to batch size of 5
        log_prob = log_prob.sum(-1, keepdim=True)

        return log_prob
