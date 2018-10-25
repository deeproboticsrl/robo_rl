import torch
from robo_rl.common import LinearGaussianNetwork
from torch.distributions import Normal


class LinearGaussianEncoder(LinearGaussianNetwork):

    def __init__(self, layers_size, final_layer_function, activation_function, log_std_min=-20, log_std_max=2):
        super().__init__(layers_size=layers_size, final_layer_function=final_layer_function,
                         activation_function=activation_function, is_layer_norm=False)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def sample(self, x):
        mean, log_std = self.forward(x)

        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        std = torch.exp(log_std)

        normal = Normal(mean, std)
        z = normal.rsample()  # reparameterization trick

        return z
