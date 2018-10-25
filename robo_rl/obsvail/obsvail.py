import pickle

import numpy as np
import torch
from pros_ai import get_policy_observation, get_expert_observation
from robo_rl.common import TrajectoryBuffer, xavier_initialisation


class ObsVAIL:

    def __init__(self, expert_file_path, discriminator, encoder, off_policy_algorithm, env, absorbing_state_dim,
                 beta_init, optimizer, beta_lr, discriminator_lr, encoder_lr, context_dim,
                 writer, weight_decay=0, grad_clip=0.01, loss_clip=100, batch_size=16,
                 clip_val_grad=False, clip_val_loss=False, replay_buffer_capacity=100000,
                 learning_rate_decay=0.5, learning_rate_decay_training_steps=1e5,
                 discriminator_weight_decay=0.001, encoder_weight_decay=0.01
                 ):

        self.discriminator = discriminator
        self.encoder = encoder
        self.off_policy_algorithm = off_policy_algorithm
        self.current_iteration = 1
        self.env = env
        self.context_dim = context_dim
        self.writer = writer
        self.batch_size = batch_size

        # Load expert trajectories
        with open(expert_file_path, "rb") as expert_file:
            expert_trajectories = pickle.load(expert_file)

        # Trajectory Length = Expert trajectory length + 2 (for absorbing states)
        self.trajectory_length = len(expert_trajectories[0]["trajectory"]) + 2

        # Wrap expert trajectories
        self._wrap_trajectories(expert_trajectories, add_absorbing=True)

        # Fill expert buffer
        self.expert_buffer = TrajectoryBuffer(capacity=len(expert_trajectories))
        for expert_trajectory in expert_trajectories:
            self.expert_buffer.add(expert_trajectory)

        # initialise replay buffer
        self.replay_buffer = TrajectoryBuffer(capacity=replay_buffer_capacity)

        observation = self.env.reset(project=False)
        self.policy_state_dim = get_policy_observation(observation).shape[0]
        self.expert_state_dim = get_expert_observation(observation).shape[0]

        # Absorbing state has last(indicator) dimension as 1 and all others as 0.
        absorbing_state_temp = [0] * absorbing_state_dim
        absorbing_state_temp[-1] = 1
        self.absorbing_state = np.array(absorbing_state_temp)

        self.discriminator_weight_decay = discriminator_weight_decay
        self.encoder_weight_decay = encoder_weight_decay
        self.grad_clip = grad_clip
        self.loss_clip = loss_clip
        self.clip_val_grad = clip_val_grad
        self.clip_val_loss = clip_val_loss

        self.beta_init = beta_init
        self.beta_lr = beta_lr
        self.discriminator_lr = discriminator_lr
        self.encoder_lr = encoder_lr

        # initialise parameters
        self.beta = self.beta_init
        self.discriminator.apply(xavier_initialisation)
        self.encoder.apply(xavier_initialisation)

        # initialise optimisers
        self.discriminator_optimizer = optimizer(self.discriminator.parameters(), lr=self.discriminator_lr,
                                                 weight_decay=self.discriminator_weight_decay)
        self.encoder_optimizer = optimizer(self.encoder.parameters(), lr=self.encoder_lr,
                                           weight_decay=self.encoder_weight_decay)

    def train(self, save_iter, num_iterations=1000):

        for iteration in range(self.current_iteration, self.current_iteration + num_iterations + 1):

            # Sample trajectory using policy

            policy_trajectory = []
            observation = get_policy_observation(self.env.reset(project=False))

            # Sample random context for the trajectory
            context = [np.random.randint(0, 1) for _ in range(self.context_dim)]

            state = torch.Tensor(np.append(observation, context))
            done = False
            timestep = 0

            # Episode reward is used only as a metric for performance
            episode_reward = 0
            while not done and timestep <= self.trajectory_length - 2:
                action = self.off_policy_algorithm.get_action(state).detach()
                observation, reward, done, _ = self.env.step(np.array(action), project=False)
                observation = get_policy_observation(observation)
                sample = dict(state=observation, action=action, reward=reward, is_absorbing=False)
                policy_trajectory.append(sample)

                state = torch.Tensor(np.append(observation, context))
                episode_reward += reward
                timestep += 1

            # Wrap policy trajectory with absorbing state and store in replay buffer
            policy_trajectory = {"trajectory": policy_trajectory, "context": context}
            self._wrap_trajectories([policy_trajectory])
            self.replay_buffer.add(policy_trajectory)

            self.writer.add_scalar("Episode reward", episode_reward, global_step=iteration)

            # TODO update D E beta and pi
            """ For each timestep, sample a mini-batch from both expert and replay buffer.
            Use it to update discriminator, encoder, policy, and beta
            """
            for timestep in range(self.trajectory_length):

                phase = min(1.0, timestep / (self.trajectory_length - 2))

                """ Expert observations are dictionary containing state, context and absorbing state indicator
                Replay observations additionally have action and RL reward from the environment
                """
                expert_batch = self.expert_buffer.sample_timestep(batch_size=self.batch_size, timestep=timestep)
                replay_batch = self.replay_buffer.sample_timestep(batch_size=self.batch_size, timestep=timestep)

                # Encode the states
                for i in range(self.batch_size):
                    if not expert_batch[i]["is_absorbing"]:
                        expert_batch[i]["encoded_state"] = self.encoder(torch.Tensor(expert_batch[i]["state"]))
                    if not replay_batch[i]["is_absorbing"]:
                        replay_batch[i]["encoded_state"] = self.encoder(torch.Tensor(replay_batch[i]["state"]))

                # Prepare observation for discriminator

                # Calculate losses and rewards

                # Prepare batch for off policy update

            # TODO save
            self.current_iteration += 1

    def _wrap_trajectories(self, trajectories, add_absorbing=False):
        """Wrap trajectories with absorbing state transition.
        Assumed each transition in trajectory to be a dict which can contain the following
        State, Action, Environment Reward, Context, Abosrbing state indicator
        If add_absorbing is True, then absorbing state indicator is added to each state in the trajectory
        else it is assumed to be already present and only absorbing transition is added."""

        for trajectory in trajectories:
            if add_absorbing:
                for timestep in range(len(trajectory["trajectory"])):
                    trajectory["trajectory"][timestep]["is_absorbing"] = False

            # Pad trajectory with absorbing state
            for i in range(len(trajectory["trajectory"]), self.trajectory_length):
                trajectory["trajectory"].append({"is_absorbing": True})
