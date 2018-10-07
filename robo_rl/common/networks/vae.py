from robo_rl.common.networks.linear_network import LinearNetwork
from robo_rl.common.networks.linear_network import LinearGaussianNetwork
import torch.nn as nn
import torch.nn.functional as torchfunc
from robo_rl.common.utils.nn_utils import no_activation
from torch.distributions import Normal


class LinearVAE(nn.Module):

    def __init__(self, encoder_dim, bottleneck_size, decoder_dim):
        """
        Example
        encoder_dim = [15,5]
        decoder_dim = [5,15]
        bottleneck_size = 2
        Then encoder is [15,5,2] and outputs mean and std each of size 2
        Decoder is [2,5,15]
        """
        super().__init__()
        encoder_layers_size = encoder_dim.append(bottleneck_size)
        decoder_layers_size = [bottleneck_size]
        decoder_layers_size.extend(decoder_dim)

        self.encoder = LinearGaussianNetwork(encoder_layers_size)
        self.decoder = LinearNetwork(decoder_layers_size)

    def encode(self, x, final_layer_function=no_activation, activation_function=torchfunc.elu):
        return self.encoder.forward(x, final_layer_function, activation_function)

    def decode(self, z, final_layer_function=no_activation, activation_function=torchfunc.elu):
        return self.decoder.forward(z, final_layer_function, activation_function)

    def forward(self, x, final_layer_function=no_activation, activation_function=torchfunc.elu):
        mean, log_std = self.encode(x, final_layer_function, activation_function)
        std = log_std.exp()

        normal = Normal(mean, std)
        if self.training:
            z = normal.rsample()    # reparameterization trick
        else:
            z = mean

        return self.decode(z), mean, log_std

