from robo_rl.common import Buffer


class ObsVAIL:

    def __init__(self, expert_buffer, discriminator, encoder, off_policy_algo, env, replay_buffer_capacity=100000):
        """policy should also expose it's replay buffer to allow adding absorbing state transitions"""
        self.expert_buffer = expert_buffer
        self.discriminator = discriminator
        self.encoder = encoder
        self.off_policy_algo = off_policy_algo
        self.current_iteration = 1
        self.env = env

        # initialise replay buffer
        self.replay_buffer = Buffer(capacity=replay_buffer_capacity)

        # TODO absorbing state implemented by adding 1 more dimension(a bool) to state space

        # TODO wrap all expert trajectories with absorbing state and make them of equal length .
        """Then D can judge whether to go here or not based on expert and assign reward. 
        so this  is how reward for absorbing state is learnt 
        """

    def train(self, num_iterations=100, learning_rate=1e-3, learning_rate_decay=0.5,
              learning_rate_decay_training_steps=1e5):

        for iteration in range(self.current_iteration, self.current_iteration + num_iterations + 1):

            # TODO sample trajectory from sac policy
            # TODO wrap policy trajectory with absorbing state
            end_trjectory_bool = False
            while end_trjectory_bool:
                """In each trajectory
                First sample an expert trajectory,then an initial state from it.
                Set this state forcefully in env. How???
                Then run trajectory to match expert trajectory length.
                For each trajectory generated, will need to store start time in replay buffer too.
                Should we pad with a startsourcing state?
                Also if env done occurs then pad with absorbing state.
                
                2 absorbing states - Good expert padding 
                and environment termination badding
                """

                # for i in len(trajectory):
                """why this for loop"""
                # TODO sample mini batches from replay buffer and expert buffer
                """How to have mini batches at same format.
                 Might have to sample 1 at a time only
                """
                # TODO Calculate loss for discriminator using above sample and update it
                """ In VAIL's case will need to add mutual info terms too
                """

            # for i in len(trajectory):
            # TODO sample mini batches from replay buffer
            # TODO Calculate reward for policy using above sample and discriminator
            # TODO Update policy using DDPG + TD3 and the batch sampled above

        pass
