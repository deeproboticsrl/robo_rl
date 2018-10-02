
class ObsGAIL:

    def __init__(self, expert_buffer, discriminator, policy, env):
        """policy should also expose it's replay buffer to allow adding absorbing state transitions"""
        self.expert_buffer = expert_buffer
        self.discriminator = discriminator
        self.policy = policy
        self.env = env
        self.current_iteration = 1

    def train(self, num_iterations=100, learning_rate=1e-3, learning_rate_decay=0.5,
              learning_rate_decay_training_steps=1e5):

        for iteration in range(self.current_iteration,self.current_iteration+num_iterations):
            a = 5