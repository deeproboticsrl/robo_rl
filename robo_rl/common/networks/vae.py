from robo_rl.common import LinearNetwork, LinearGaussianNetwork
import torch.nn as nn
import torch.nn.functional as torchfunc
from robo_rl.common.utils.nn_utils import no_activation
from torch.distributions import Normal


class LinearVAE(nn.Module):

    def __init__(self, encoder_dim, bottleneck_size, decoder_dim, final_layer_function=no_activation,
                 activation_function=torchfunc.elu):
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

        self.encoder = LinearGaussianNetwork(encoder_layers_size,final_layer_function=final_layer_function,
                                             activation_function=activation_function)
        self.decoder = LinearNetwork(decoder_layers_size,final_layer_function=final_layer_function,
                                     activation_function=activation_function)

    def encode(self, x):
        return self.encoder.forward(x)

    def decode(self, z):
        return self.decoder.forward(z)

    def forward(self, x):
        mean, log_std = self.encode(x)
        std = log_std.exp()

        normal = Normal(mean, std)
        if self.training:
            z = normal.rsample()    # reparameterization trick
        else:
            z = mean

        return self.decode(z), mean, log_std

