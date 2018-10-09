from robo_rl.common.networks import LinearNetwork


class LinearDiscriminator(LinearNetwork):

    def __init__(self, input_dim, hidden_dim):
        """

        Example If hidden_dim = [200,300]
        Number of input = 10
        Number of hidden layers = 2 with 200 nodes in 1st layer and 300 in next layer
        Number of outputs = 1  (probability that input is fake)
        """
        layers_size = [input_dim]
        layers_size.extend(hidden_dim)
        super().__init__(layers_size=layers_size)

    def forward(self, x):
        return super().forward(x)
