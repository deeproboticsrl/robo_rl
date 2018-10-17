import torch
import torch.nn.functional as torchfunc
from robo_rl.common import LinearNetwork, LinearPFNN


class LinearDiscriminator(LinearNetwork):

    def __init__(self, input_dim, hidden_dim, activation_function=torchfunc.relu, bias=True):
        """

        Example If hidden_dim = [200,300]
        Number of input = 10
        Number of hidden layers = 2 with 200 nodes in 1st layer and 300 in next layer
        Number of outputs = 1  (probability that input is fake)
        """
        layers_size = [input_dim]
        layers_size.extend(hidden_dim)
        layers_size.append(1)
        super().__init__(layers_size=layers_size, is_layer_norm=False, activation_function=activation_function,
                         final_layer_function=torch.sigmoid, bias=bias)

    def forward(self, x):
        return super().forward(x)


class LinearPFDiscriminator(LinearPFNN):

    def __init__(self, input_dim, hidden_dim, activation_function=torchfunc.elu, bias=True, num_networks=5):
        layers_size = [input_dim]
        layers_size.extend(hidden_dim)
        layers_size.append(1)
        super().__init__(layers_size=layers_size, activation_function=activation_function, num_networks=num_networks,
                         final_layer_function=torch.sigmoid, bias=bias)

    def forward(self, x):
        return super().forward(x)
