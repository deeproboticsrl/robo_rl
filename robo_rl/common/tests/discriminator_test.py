import robo_rl.common.utils.nn_utils as nn_utils
import robo_rl.common.utils.utils as utils
import torch
from robo_rl.common.networks.discriminator import LinearDiscriminator
from torchviz import make_dot

input = 2
hidden_sizes = [4, 3]
discriminator = LinearDiscriminator(input_dim=input, hidden_dim=hidden_sizes)

utils.print_heading("Network architecture")
print(discriminator)

# batch of inputs are supported implicitly in forward
x = torch.Tensor([1, 2])
y = discriminator.forward(x)

discriminator.apply(nn_utils.xavier_initialisation)

utils.print_heading("Check output shape")
print(x, y)

y.backward()

utils.print_heading("Weights and grads of final layer")
print((discriminator.linear_layers[2]).weight)
print((discriminator.linear_layers[2]).weight.grad)

dot = make_dot(y)
dot.render()
