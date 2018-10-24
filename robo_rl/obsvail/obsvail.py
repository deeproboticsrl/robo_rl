import pickle

import numpy as np
from pros_ai import get_policy_observation, get_expert_observation
from robo_rl.common import TrajectoryBuffer


class ObsVAIL:

    def __init__(self, expert_file_path, discriminator, encoder, off_policy_algorithm, env, absorbing_state_dim,
                 replay_buffer_capacity=100000):

        # Load expert trajectories
        with open(expert_file_path, "rb") as expert_file:
            expert_trajectories = pickle.load(expert_file)

        # Trajectory Length = Expert trajectory length + 2 (for absorbing states)
        self.trajectory_length = len(expert_trajectories[0]) + 2

        # Wrap expert trajectories
        self._wrap_trajectories(expert_trajectories, add_absorbing=True)

        # Fill expert buffer
        self.expert_buffer = TrajectoryBuffer(capacity=len(expert_trajectories))
        for expert_trajectory in expert_trajectories:
            self.expert_buffer.add(expert_trajectory)

        self.discriminator = discriminator
        self.encoder = encoder
        self.off_policy_algorithm = off_policy_algorithm
        self.current_iteration = 1
        self.env = env

        # initialise replay buffer
        self.replay_buffer = TrajectoryBuffer(capacity=replay_buffer_capacity)

        observation = self.env.reset(project=False)
        self.policy_state_dim = get_policy_observation(observation).shape[0]
        self.expert_state_dim = get_expert_observation(observation).shape[0]

        # Absorbing state has last(indicator) dimension as 1 and all others as 0.
        absorbing_state_temp = [0] * absorbing_state_dim
        absorbing_state_temp[-1] = 1
        self.absorbing_state = np.array(absorbing_state_temp)

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

    def _wrap_trajectories(self, trajectories, add_absorbing=False):
        """Wrap trajectories with absorbing state transition.
        Assumed each transition in trajectory to be a dict which can contain the following
        State, Action, Environment Reward, Context, Abosrbing state indicator
        If add_absorbing is True, then absorbing state indicator is added to each state in the trajectory
        else it is assumed to be already present and only absorbing transition is added."""

        for trajectory in trajectories:
            if add_absorbing:
                for timestep in range(len(trajectory)):
                    trajectory[timestep]["is_absorbing"] = False

            # Assumed same context for whole trajectory
            trajectory_context = trajectory[0]["context"]

            # Pad trajectory with absorbing state
            for i in range(len(trajectory), self.trajectory_length):
                trajectory.append({"is_absorbing": True, "context": trajectory_context})
