import torch
import torch.nn.functional as torchfunc
from robo_rl.common.networks.linear_network import LinearNetwork


class ValueNetwork(LinearNetwork):

    def __init__(self, state_dim, hidden_dim):
        layers_size = [state_dim]
        layers_size.extend(hidden_dim)
        layers_size.append(1)
        super().__init__(layers_size)

    def forward(self, state):
        return super().forward(state, torchfunc.linear, torchfunc.elu)


class QNetwork(LinearNetwork):

    def __init__(self, state_dim, action_dim, hidden_dim):
        layers_size = [state_dim + action_dim]
        layers_size.extend(hidden_dim)
        layers_size.append(1)
        super().__init__(layers_size)

    def forward (self, state, action):
        x = torch.cat([state, action], 1)
        return super().forward(x, torchfunc.linear, torchfunc.elu)




