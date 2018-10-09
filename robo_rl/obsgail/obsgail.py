from robo_rl.common import Buffer


class ObsGAIL:

    def __init__(self, expert_buffer, discriminator, off_policy_algo, replay_buffer_capacity=100000):
        """policy should also expose it's replay buffer to allow adding absorbing state transitions"""
        self.expert_buffer = expert_buffer
        self.discriminator = discriminator
        self.off_policy_algo = off_policy_algo
        self.current_iteration = 1

        # initialise replay buffer
        self.replay_buffer = Buffer(capacity=replay_buffer_capacity)

        # TODO absorbing state implemented by adding 1 more dimension(a bool) to state space

        # TODO wrap all expert trajectories with absorbing state .
        """Then D can judge whether to go here or not based on expert and assign reward. 
        so this  is how reward for absorbing state is learnt 
        """

    def train(self, num_iterations=100, learning_rate=1e-3, learning_rate_decay=0.5,
              learning_rate_decay_training_steps=1e5):

        for iteration in range(self.current_iteration,self.current_iteration+num_iterations+1):

            # TODO sample trajectory from sac policy

            # TODO wrap policy trajectory with absorbing state

            # for i in len(trajectory):
                """why this for loop"""
                # TODO sample mini batches from replay buffer and expert buffer
                """Sampled at some random offset 
                """
                # TODO Calculate loss for discriminator using above sample and update it

            # for i in len(trajectory):
                # TODO sample mini batche from replay buffer
                # TODO Calculate reward for policy using above sample and discriminator
                # TODO Update policy using SAC + TD3 using batch sampled above

        # TODO Remember return for final states will use reward for absorbing state which is learnt

        pass
