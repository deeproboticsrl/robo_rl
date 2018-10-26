import torch
import torch.nn as nn
import torch.nn.functional as torchfunc
from robo_rl.common import LinearNetwork


class LinearPFNN(nn.Module):
    """Phase Functioned Neural Network
    Linear Networks are used.
    Linear Interpolation done between weights in a phase interval.

    Refer https://xbpeng.github.io/projects/VDB/2018_VDB.pdf

    The phase is a value between 0 to 1. 0 & 1 being equal i.e. it is cyclic
    [0,1] is divided into num_networks equally spaced intervals and
    each interval endpoint has a neural net associated with it.
    The weights for a phase is linear interpolation of weights of endpoints
    of the interval in which the phase lies.
    """

    def __init__(self, layers_size, final_layer_function, activation_function, num_networks=5, bias=True):
        self.num_networks = num_networks
        self.bias = bias
        self.activation_function = activation_function
        self.final_layer_function = final_layer_function
        super().__init__()
        self.basis_networks = nn.ModuleList(
            [LinearNetwork(layers_size=layers_size, final_layer_function=final_layer_function,
                           activation_function=activation_function, is_layer_norm=False,
                           is_dropout=False, bias=bias)
             for _ in range(num_networks)])

    def forward(self, x):
        """Expected x to be dict containing input tensor and it's phase
        Batch operations not supported yet since for different phases, already using different nets(weights)
        so forward is called individually for them.

        NOTE - USE None_grad function instead of zero_grad if using an optimiser which uses history to make grad updates
        """
        input_tensor = x["input"]

        phase = x["phase"]
        # Enforce phase in [0,1)
        phase = phase - int(phase)

        # Get indices for interval (in which input phase belongs) endpoints
        left_index = int(phase * self.num_networks) % self.num_networks
        right_index = (left_index + 1) % self.num_networks

        # Calculate phase for interval endpoints
        left_phase = left_index / self.num_networks
        right_phase = left_phase + (1 / self.num_networks)

        # phase = weight * left_phase + (1-weight) * right_phase
        interpolation_weight = (right_phase - phase) * self.num_networks

        left_net = self.basis_networks[left_index]
        right_net = self.basis_networks[right_index]

        x = input_tensor

        # Forward pass of the underlying net using functional API
        for i in range(len(left_net.linear_layers) - 1):
            weight = interpolation_weight * torch.Tensor(left_net.linear_layers[i].weight) + (
                    1 - interpolation_weight) * torch.Tensor(right_net.linear_layers[i].weight)
            if self.bias:
                bias = interpolation_weight * torch.Tensor(left_net.linear_layers[i].bias) + (
                        1 - interpolation_weight) * torch.Tensor(right_net.linear_layers[i].bias)
                x = torchfunc.linear(x, weight=weight, bias=bias)
            else:
                x = torchfunc.linear(x, weight=weight)
            x = self.activation_function(x)

        final_layer_weight = interpolation_weight * torch.Tensor(left_net.linear_layers[-1].weight) + (
                1 - interpolation_weight) * torch.Tensor(right_net.linear_layers[-1].weight)

        if self.bias:
            final_layer_bias = interpolation_weight * torch.Tensor(left_net.linear_layers[-1].bias) + (
                    1 - interpolation_weight) * torch.Tensor(right_net.linear_layers[-1].bias)
            x = torchfunc.linear(x, weight=final_layer_weight, bias=final_layer_bias)
        else:
            x = torchfunc.linear(x, weight=final_layer_weight)

        return self.final_layer_function(x)

