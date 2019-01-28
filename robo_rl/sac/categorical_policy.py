from robo_rl.common.networks.linear_network import LinearCategoricalNetwork
import torch.nn.functional as torchfunc
import torch




class LinearCategoricalPolicy (LinearCategoricalNetwork):
    def __init__(self, state_dim, action_dim, hidden_dim, is_layer_norm=False):
        # action dim =[3,4,2,5]:
        # 1st variable has 3 possible action
        # 2nd variable has 4 possible action
        layers_size = [state_dim]
        layers_size.extend(hidden_dim)
        layers_size.append(action_dim)
        super().__init__(layers_size=layers_size, is_layer_norm=is_layer_norm,
                         activation_function=torchfunc.relu)
    def forward(self,state):
        return super().forward(state)


    def get_action(self,state):
        categories = self.forward(state)
        sampled_action= [x.sample() for x in categories]
        return sampled_action





