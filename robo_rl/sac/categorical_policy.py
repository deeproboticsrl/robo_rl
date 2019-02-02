import torch
import torch.nn.functional as torchfunc
from robo_rl.common.networks.linear_network import LinearCategoricalNetwork
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform


def gumbel_sample(shape):
    dist = Uniform(torch.Tensor([0]), torch.Tensor([1]))
    u = dist.sample(sample_shape=shape)
    g = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    return g.squeeze()


class LinearCategoricalPolicy(LinearCategoricalNetwork):
    def __init__(self, state_dim, action_dim, hidden_dim, is_layer_norm=False):
        # action dim =[3,4,2,5]:
        # 1st variable has 3 possible action
        # 2nd variable has 4 possible action
        self.action_dim = action_dim
        layers_size = [state_dim]
        layers_size.extend(hidden_dim)
        layers_size.append(action_dim)
        super().__init__(layers_size=layers_size, is_layer_norm=is_layer_norm,
                         activation_function=torchfunc.relu)

    def forward(self, state):
        return super().forward(state)

    # https://arxiv.org/pdf/1611.01144.pdf  CATEGORICAL REPARAMETERIZATION
    #  WITH GUMBEL-SOFTMAX
    # z= (log(πi) + gi)/τ
    def get_action(self, state, softmax_temperature=0.1):
        # (batch size, state dim)
        assert len(state.shape) == 2
        pi = self.forward(state)
        c = 0
        categorical_obj = []
        for ith_action_dim in self.action_dim:
            sliced_pi = pi[:, c:c + ith_action_dim]
            z = torch.add(torch.log(sliced_pi), gumbel_sample(sliced_pi.size()))
            z = z / softmax_temperature
            soft_z = torchfunc.softmax(z,dim=1)
            categorical_obj.append(Categorical(soft_z))  # categorical obj for sampling actions
            c += ith_action_dim
        sampled_action = torch.Tensor([x.sample() for x in categorical_obj])
        log_prob = torch.Tensor([x.log_prob(sampled_action) for x in categorical_obj])

        # raise SystemExit
        return sampled_action, log_prob
