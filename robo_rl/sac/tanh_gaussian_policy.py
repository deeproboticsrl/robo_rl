import torch
import torch.nn.functional as torchfunc
from torch.distributions import Normal
from robo_rl.common.networks.linear_network import LinearGaussianNetwork

## GMM policy L to be implemented
# hidden_dim is now array of hidden layers e.g =[20,30] First hidden layer has 20 units and 2nd has 30 units

class TanhGaussianPolicy(LinearGaussianNetwork):

    def __init__(self, state_dim, action_dim, hidden_dim, layer_norm=True):

        layers_size = [state_dim]
        layers_size.extend(hidden_dim)
        layers_size.append(action_dim)
        super().__init__(layers_size, layer_norm)

    def forward(self, state):
        return super().forward(state, final_layer_function=torchfunc.linear, activation_function=torchfunc.elu)

    def evaluation(self, state, epsilon=1e-6, reparameterize=True, deterministic=False):
        mean, log_std = self.forward(state)
        if deterministic:
            return mean

        std = log_std.exp()

        normal = Normal(mean, std)

        if reparametrize:
            z = normal.rsample()    # reparameterization trick
        else:
            z = normal.sample()

        action = torch.tanh(z)
        # return action
        log_prob =  normal.log_prob(z)-torch.log(1-action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)  # if tensor is 4*4 row sum wil be 4*1
        return action, z, log_prob, mean, log_std

        # Action bound  squash correction

    def get_action(self,state,reparameterize=True,deterministic=False):
        mean,log_std =self.forward(state)
        if deterministic:
            return mean
        std = log_std.exp()

        normal=Normal(mean,std)
        if reparameterize:
            z=normal.rsample()
        else:
            z=normal.sample()
        action = torch.tanh(z)

        return action




