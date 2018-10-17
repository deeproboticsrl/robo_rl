import torch.nn as nn


class LinearNetwork(nn.Module):

    def __init__(self, layers_size, final_layer_function, activation_function, is_layer_norm=True,
                 is_final_layer_norm=False, is_dropout=False, dropout_probability=0.5, bias=True,
                 requires_grad=True):
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

        self.is_layer_norm = is_layer_norm
        self.is_final_layer_norm = is_final_layer_norm
        self.is_dropout = is_dropout

        super().__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(layers_size[i], layers_size[i + 1], bias=bias,
                                                      requires_grad=requires_grad)
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

    def __init__(self, layers_size, final_layer_function, activation_function, is_layer_norm=True):
        """
        Example If layers_size = [10,200,30,3]
        Number of inputs = 10
        Number of hidden layers = 2 with 200 nodes in 1st layer and 300 in next layer
        Number of outputs = 3 ---- output layer has 2 units 1 for mean and other for log_std
        """
        super().__init__(layers_size[:-1], final_layer_function, activation_function, is_layer_norm,
                         is_final_layer_norm=True)
        self.mean_layer = nn.Linear(layers_size[-2], layers_size[-1])
        self.log_std_layer = nn.Linear(layers_size[-2], layers_size[-1])

    def forward(self, x):
        x = super().forward(x)
        mean = self.final_layer_function(self.mean_layer(x))
        log_std = self.final_layer_function(self.log_std_layer(x))
        return mean, log_std
