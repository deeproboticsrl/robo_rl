import torch
import torch.nn as nn
import torch.nn.functional as torchfunc
from robo_rl.common.utils.nn_utils import no_activation
from torch.distributions.categorical import Categorical


class LinearNetwork(nn.Module):

    def __init__(self, layers_size, final_layer_function, activation_function, is_layer_norm=False,
                 is_final_layer_norm=False, is_dropout=False, dropout_probability=0.5, bias=False):
        """
        Arguments:

            layers_size (list/array of integers) : number of nodes in each layer of the neural network
            including input and output

            final_layer_function (pytorch function): activation function to be applied at final layer of network

            activation_function (pytorch function): activation function to be applied at each hidden layer

            is_layer_norm (bool, optional): whether to apply layer norm to each layer in the network

            is_final_layer_norm (bool, optional): whether to apply layer norm to final(output) layer.
            Use this only when this network's output will be cascaded to further networks

        Example If layers_size = [10,200,300,2]
        Number of inputs = 10
        Number of hidden layers = 2 with 200 nodes in 1st layer and 300 in next layer
        Number of outputs = 2
        """
        self.final_layer_function = final_layer_function
        self.activation_function = activation_function

        self.bias = bias
        self.is_layer_norm = is_layer_norm
        self.is_final_layer_norm = is_final_layer_norm
        self.is_dropout = is_dropout

        super().__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(layers_size[i], layers_size[i + 1], bias=bias)
                                            for i in range(len(layers_size) - 1)])
        if self.is_layer_norm:
            # applied on outputs of hidden layers
            self.layer_norm_layers = nn.ModuleList([nn.LayerNorm(layers_size[i])
                                                    for i in range(1, len(layers_size) - 1)])
            if self.is_final_layer_norm:
                self.layer_norm_layers.append(nn.LayerNorm(layers_size[-1]))

        if self.is_dropout:
            self.dropout_layer = nn.AlphaDropout(p=dropout_probability)

    def forward(self, x):
        for i in range(len(self.linear_layers) - 1):
            x = self.linear_layers[i](x)
            if self.is_layer_norm:
                x = self.layer_norm_layers[i](x)
            x = self.activation_function(x)
            if self.is_dropout:
                x = self.dropout_layer(x)

        x = self.linear_layers[-1](x)
        if self.is_layer_norm and self.is_final_layer_norm:
            x = self.layer_norm_layers[-1](x)
        return self.final_layer_function(x)


class LinearGaussianNetwork(LinearNetwork):

    def __init__(self, layers_size, final_layer_function, activation_function, is_layer_norm=False):
        """
        Example If layers_size = [10,200,30,3]
        Number of inputs = 10
        Number of hidden layers = 2 with 200 nodes in 1st layer and 300 in next layer
        Number of outputs = 3 ---- output layer has 2 units 1 for mean and other for log_std
        """
        super().__init__(layers_size[:-1], final_layer_function, activation_function, is_layer_norm,
                         is_final_layer_norm=is_layer_norm)
        self.mean_layer = nn.Linear(layers_size[-2], layers_size[-1], bias=False)
        self.log_std_layer = nn.Linear(layers_size[-2], layers_size[-1], bias=False)

    def forward(self, x):
        x = super().forward(x)
        mean = self.final_layer_function(self.mean_layer(x))
        log_std = self.final_layer_function(self.log_std_layer(x))
        return mean, log_std


class LinearCategoricalNetwork(LinearNetwork):
    def __init__(self, layers_size, activation_function, is_layer_norm=False):
        """
        Example If layers_size = [10,200,30,[3,3,3]]
        Number of inputs = 10
        Number of hidden layers = 2 with 200 nodes in 1st layer and 300 in next layer
        Number of outputs = 3 ---- output layer has 2 units 1 for mean and other for log_std
        """
        super().__init__(layers_size[:-1], final_layer_function=no_activation, activation_function=activation_function,
                         is_layer_norm=is_layer_norm, is_final_layer_norm=is_layer_norm)
        self.output_layer = nn.Linear(layers_size[-2], sum(layers_size[-1]))
        self.final_layer_dim = layers_size[-1]

    def forward(self, x):
        x = super().forward(x)
        x = self.output_layer(x)
        c = 0
        y = torch.Tensor(x.size())
        # print(y.size())
        for ith_action_dim in self.final_layer_dim:
            # Normalisation for probability distribution
            # print(torch.sum(x[:, c:c + ith_action_dim], dim=1))
            # print(normalisation_constant.size(),x[:, c: c + ith_action_dim].size())
            y[:, c: c + ith_action_dim] = torchfunc.softmax(x[:, c: c + ith_action_dim], dim=1)
            c += ith_action_dim
        return y

    def sampled_forward(self, x):
        x = super().forward(x)
        x = self.output_layer(x)
        c = 0
        y = torch.Tensor(x.size())
        categorical_obj = []
        for ith_action_dim in self.final_layer_dim:
            y[c: c + ith_action_dim] = torchfunc.softmax(x[c: c + ith_action_dim], dim=0)
            categorical_obj.append(Categorical(y[c: c + ith_action_dim]))
            c += ith_action_dim
        return categorical_obj
