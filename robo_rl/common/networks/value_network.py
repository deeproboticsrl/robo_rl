import torch
import torch.nn.functional as torchfunc
from robo_rl.common.networks.linear_network import LinearNetwork


class LinearValueNetwork(LinearNetwork):

    def __init__(self, state_dim, hidden_dim, layer_norm=True):
        layers_size = [state_dim]
        layers_size.extend(hidden_dim)
        layers_size.append(1)
        super().__init__(layers_size, layer_norm)

    def forward(self, state):
        return super().forward(state, final_layer_function=torchfunc.linear, activation_function=torchfunc.elu)


class LinearQNetwork(LinearNetwork):

    def __init__(self, state_dim, action_dim, hidden_dim, layer_norm=True):
        layers_size = [state_dim + action_dim]
        layers_size.extend(hidden_dim)
        layers_size.append(1)
        super().__init__(layers_size, layer_norm)

    def forward (self, state, action):
        x = torch.cat([state, action], 1)
        return super().forward(x, final_layer_function=torchfunc.linear, activation_function=torchfunc.elu)




