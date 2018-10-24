import pickle

import numpy as np
import torch
from pros_ai import get_policy_observation, get_expert_observation
from robo_rl.common import TrajectoryBuffer, xavier_initialisation


class ObsVAIL:

    def __init__(self, expert_file_path, discriminator, encoder, off_policy_algorithm, env, absorbing_state_dim,
                 beta_init, optimizer, beta_lr, discriminator_lr, encoder_lr, context_dim,
                 writer, weight_decay=0, grad_clip=0.01, loss_clip=100,
                 clip_val_grad=False, clip_val_loss=False, replay_buffer_capacity=100000,
                 learning_rate_decay=0.5, learning_rate_decay_training_steps=1e5):

        self.discriminator = discriminator
        self.encoder = encoder
        self.off_policy_algorithm = off_policy_algorithm
        self.current_iteration = 1
        self.env = env
        self.context_dim = context_dim
        self.writer = writer

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

        self.weight_decay = weight_decay
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
                                                 weight_decay=self.weight_decay)
        self.encoder_optimizer = optimizer(self.encoder.parameters(), lr=self.encoder_lr,
                                           weight_decay=self.weight_decay)

    def train(self, save_iter, num_iterations=1000):

        for iteration in range(self.current_iteration, self.current_iteration + num_iterations + 1):

            # TODO sample trajectory from sac policy

            policy_trajectory = []
            observation = get_policy_observation(self.env.reset(project=False))

            # Sample random context for the trajectory
            context = [np.random.randint(0, 1) for _ in range(self.context_dim)]

            state = torch.Tensor(np.append(observation, context))
            done = False
            timestep = 0

            while not done and timestep <= self.trajectory_length - 2:
                # used only as a metric for performance
                episode_reward = 0
                action = self.off_policy_algorithm.get_action(state).detach()
                observation, reward, done, _ = gym_torchify(env.step(action))
                sample = dict(state=state, action=action, reward=reward, next_state=observation, done=done)
                buffer.add(sample)
                if len(buffer) > 10 * args.sample_batch_size:
                    for num_update in range(args.updates_per_step):
                        update_count += 1
                        batch_list_of_dicts = buffer.sample(batch_size=args.sample_batch_size)
                        batch_dict_of_lists = ld_to_dl(batch_list_of_dicts)

                        """ Combined Experience replay. Add online transition too.
                        """
                        for k in batch_list_of_dicts[0].keys():
                            batch_dict_of_lists[k].append(sample[k])
                        sac.policy_update(batch_dict_of_lists, update_number=update_count)

                episode_reward += reward
                state = observation
                timestep += 1

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
                for timestep in range(len(trajectory["trajectory"])):
                    trajectory["trajectory"][timestep]["is_absorbing"] = False

            # Pad trajectory with absorbing state
            for i in range(len(trajectory["trajectory"]), self.trajectory_length):
                trajectory["trajectory"].append({"is_absorbing": True})
