import torch.nn as nn
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
        super().__init__()
        self.basis_networks = nn.ModuleList(
            [LinearNetwork(layers_size=layers_size, final_layer_function=final_layer_function,
                           activation_function=activation_function, is_layer_norm=False,
                           is_dropout=False, bias=bias)
             for _ in range(num_networks)])
        # This network is used for forward
        self.main_network = LinearNetwork(layers_size=layers_size, final_layer_function=final_layer_function,
                                          activation_function=activation_function, is_layer_norm=False,
                                          is_dropout=False, bias=bias,requires_grad=False)

    def forward(self, x):
        """Expected x to be dict containing input tensor and it's phase
        Batch operations not supported yet since for different phases, already using different nets(weights)
        so forward is called individually for them.
        """
        input_tensor = x["input"]

        phase = x["phase"]
        # Enforce phase in [0,1)
        phase = phase - int(phase)

        # Get indices for interval endpoints
        left_index = int(phase * self.num_networks) % self.num_networks
        right_index = (left_index + 1) % self.num_networks

        left_phase = left_index / self.num_networks
        right_phase = left_phase + (1 / self.num_networks)

        # phase = weight * left_phase + (1-weight) * right_phase
        weight = (right_phase - phase) * self.num_networks

        left_net = self.basis_networks[left_index]
        right_net = self.basis_networks[right_index]

        for main_param, left_param, right_param in zip(self.main_network.parameters(), left_net.parameters(),
                                                       right_net.parameters()):
            main_param.copy_(weight * left_param + (1 - weight) * right_param)

        return self.main_network.forward(input_tensor)
