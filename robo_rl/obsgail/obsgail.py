from robo_rl.common import Buffer


class ObsGAIL:

    def __init__(self, expert_buffer, discriminator, policy, env, replay_buffer_capacity=100000):
        """policy should also expose it's replay buffer to allow adding absorbing state transitions"""
        self.expert_buffer = expert_buffer
        self.discriminator = discriminator
        self.policy = policy
        self.env = env
        self.current_iteration = 1

        # initialise replay buffer
        self.replay_buffer = Buffer(capacity=replay_buffer_capacity)

        # TODO initialise absorbing state
        self.absorbing_state = None

        # TODO wrap all expert trajectories with absorbing state

    def train(self, policy, num_iterations=100, learning_rate=1e-3, learning_rate_decay=0.5,
              learning_rate_decay_training_steps=1e5):

        for iteration in range(self.current_iteration,self.current_iteration+num_iterations+1):

            # TODO sample trajectory from sac policy

            # TODO wrap policy trajectory with absorbing state

            # for i in len(trajectory):
                """why this for loop"""
                # TODO sample mini batches from replay buffer and expert buffer
                # TODO Calculate loss for discriminator using above sample and update it

            # for i in len(trajectory):
                # TODO sample mini batche from replay buffer
                # TODO Calculate reward for policy using above sample and discriminator
                # TODO Update policy using SAC + TD3 using batch sampled above

        pass
