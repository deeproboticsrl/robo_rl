import torch
import torch.nn.functional as torchfunc
from robo_rl.common.networks.linear_network import LinearGaussianNetwork
from robo_rl.common.utils.nn_utils import no_activation
from torch.distributions import Normal


# GMM policy L to be implemented
# hidden_dim is now array of hidden layers e.g =[20,30] First hidden layer has 20 units and 2nd has 30 units

class GaussianPolicy(LinearGaussianNetwork):

    def __init__(self, state_dim, action_dim, hidden_dim, layer_norm=True):

        layers_size = [state_dim]
        layers_size.extend(hidden_dim)
        layers_size.append(action_dim)
        super().__init__(layers_size, layer_norm)

    def forward(self, state):
        """ returns mean and log_std after a forward pass through a linear neural network
        """
        return super().forward(state, final_layer_function=no_activation, activation_function=torchfunc.elu)

    def get_action(self, state, squasher, epsilon=1e-6, reparameterize=True, deterministic=False,
                   log_std_min=-10, log_std_max=-1):

        mean, log_std = self.forward(state)

        # limit std. not too stochastic nor too deterministic
        log_std = torch.clamp(log_std, min=log_std_min, max=log_std_max)
        std = torch.exp(log_std)

        normal = Normal(mean, std)

        if deterministic:
            z = mean
        else:
            if reparameterize:
                z = normal.rsample()  # reparameterization trick
            else:
                z = normal.sample()

        action = squasher.squash(z)

        log_prob = normal.log_prob(z) - torch.log(squasher.derivative(z) + epsilon)

        # If tensor is 5*4, then the row sum wil be 5*1, corresponding to batch size of 5
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob
