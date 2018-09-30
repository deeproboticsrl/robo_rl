from robo_rl.obsgail.discriminator import Discriminator
import robo_rl.common.utils.utils as utils
import torch
import robo_rl.common.utils.nn_utils as nn_utils

layer_sizes = [2, 10, 10]
discriminator = Discriminator(layer_sizes)

# batch of inputs are supported implicitly in forward
x = torch.Tensor([[1, 2], [3, 4]])
y = discriminator.forward(x)

discriminator.apply(nn_utils.xavier_initialisation)

utils.print_heading("Check output shape")
print(x.numpy(), y.detach().numpy())

utils.print_heading("Network architecture")
nn_utils.print_all_modules(discriminator)
