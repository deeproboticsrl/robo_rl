import torch.nn.functional as torchfunc
import torch
from robo_rl.common.networks.linear_network import LinearNetwork


class Discriminator(LinearNetwork):

    def __init__(self,input,hidden_layer):
        """

        Example If hidden_layer = [200,300]
        Number of input = 10
        Number of hidden layers = 2 with 200 nodes in 1st layer and 300 in next layer
        Number of outputs = 1  (probability that input is fake)
        """
        layers_size=[]
        layers_size.append(input)
        layers_size.extend(hidden_layer)
        layers_size.append(1)        # for output layer
        super().__init__(layers_size=layers_size)

    def forward(self, x):
        return super().forward(x, torch.sigmoid, torchfunc.elu)






