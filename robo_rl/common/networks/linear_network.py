import torch.nn as nn


class LinearNetwork(nn.Module):

    def __init__(self, layers_size):
        """
        Example If layers_size = [10,200,300,1]
        Number of inputs = 10
        Number of hidden layers = 2 with 200 nodes in 1st layer and 300 in next layer
        Number of outputs = 1
        """
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(layers_size[i], layers_size[i+1])
                                     for i in range(len(layers_size)-1)])

    def forward(self, x, final_layer_function, activation_function):
        for layer in self.layers:
            x = activation_function(layer(x))
        return final_layer_function(x)
