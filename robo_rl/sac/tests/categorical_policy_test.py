import torch
import torch.nn as nn
import torch.nn.functional as torchfunc
from  robo_rl.common.networks.linear_network import LinearCategoricalNetwork
from robo_rl.sac.categorical_policy import LinearCategoricalPolicy
torch.manual_seed(0)
action_dim = [2, 3,4 ]
# net = LinearCategoricalNetwork([5,5,action_dim],activation_function=torchfunc.relu,is_layer_norm=False)
# print(net)
state= torch.Tensor([1,2,3,4,5])
# out =(net(state))
# print(out)
# ou= out.reshape(1,6)
# print (out.size())

policy = LinearCategoricalPolicy(5, action_dim, [5], False)
print(policy(state))
print(policy.get_action(state))
