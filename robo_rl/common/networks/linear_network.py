import torch.nn as nn


class LinearNetwork(nn.Module):

    def __init__(self, layers_size, is_layer_norm=True, is_final_layer_norm=False):
        """
        Example If layers_size = [10,200,300,2]
        Number of inputs = 10
        Number of hidden layers = 2 with 200 nodes in 1st layer and 300 in next layer
        Number of outputs = 3
        """
        self.is_layer_norm = is_layer_norm
        self.is_final_layer_norm = is_final_layer_norm
        super().__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(layers_size[i], layers_size[i + 1])
                                            for i in range(len(layers_size) - 1)])
        if self.is_layer_norm:
            # applied on outputs of hidden layers
            self.layer_norm_layers = nn.ModuleList([nn.LayerNorm(layers_size[i])
                                                    for i in range(1, len(layers_size) - 1)])
            if self.is_final_layer_norm:
                self.layer_norm_layers.append(nn.LayerNorm(layers_size[-1]))

    def forward(self, x, final_layer_function, activation_function):
        for i in range(len(self.linear_layers) - 1):
            x = self.linear_layers[i](x)
            if self.is_layer_norm:
                x = self.layer_norm_layers[i](x)
            x = activation_function(x)

        x = self.linear_layers[-1](x)
        if self.is_final_layer_norm:
            x = self.layer_norm_layers[-1](x)
        return final_layer_function(x)


class LinearGaussianNetwork(LinearNetwork):

    def __init__(self, layers_size, is_layer_norm=True):
        """
        Example If layers_size = [10,200,30,3]
        Number of inputs = 10
        Number of hidden layers = 2 with 200 nodes in 1st layer and 300 in next layer
        Number of outputs = 3 ---- output layer has 2 units 1 for mean and other for log_std
        """
        super().__init__(layers_size[:-1], is_layer_norm, is_final_layer_norm=True)
        self.mean_layer = nn.Linear(layers_size[-2], layers_size[-1])
        self.log_std_layer = nn.Linear(layers_size[-2], layers_size[-1])

    def forward(self, x, final_layer_function, activation_function):
        x = super().forward(x, activation_function, activation_function)
        mean = final_layer_function(self.mean_layer(x))
        log_std = final_layer_function(self.log_std_layer(x))
        return mean, log_std


