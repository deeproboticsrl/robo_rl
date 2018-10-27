import os
import pickle

import numpy as np
import torch
import torch.autograd as autograd
from pros_ai import get_policy_observation, get_expert_observation
from robo_rl.common import TrajectoryBuffer, xavier_initialisation, None_grad, print_heading, heading_decorator
from torch.distributions import Normal, kl_divergence
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank


class ObsVAIL:

    def __init__(self, expert_file_path, discriminator, encoder, off_policy_algorithm, env, absorbing_state_dim,
                 beta_init, optimizer, beta_lr, discriminator_lr, encoder_lr, context_dim,
                 writer, grad_clip=0.01, loss_clip=100, batch_size=16, information_constraint=0.1, gp_lambda=10,
                 clip_val_grad=False, clip_val_loss=False, replay_buffer_capacity=100000,
                 learning_rate_decay=0.5, learning_rate_decay_training_steps=1e5,
                 discriminator_weight_decay=0.001, encoder_weight_decay=0.01,
                 reward_clip=False, clip_val_reward=1):

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
        self.reward_clip = reward_clip
        self.clip_val_grad = clip_val_grad
        self.clip_val_loss = clip_val_loss
        self.clip_val_reward = clip_val_reward

        self.beta_init = beta_init
        self.beta_lr = beta_lr
        self.information_constraint = information_constraint
        self.gp_lambda = gp_lambda

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
        self.policy_update_count = 0
        self.max_reward = -np.inf

    def train(self, save_iter, modeldir, attributesdir, bufferdir, logfile, num_iterations=1000000):

        for iteration in range(self.current_iteration, self.current_iteration + num_iterations + 1):
            print(f"Starting iteration {iteration}")

            # Sample trajectory using policy

            policy_trajectory = []
            current_observation = get_policy_observation(self.env.reset(project=False))

            # Sample random context for the trajectory
            context = [np.random.randint(0, 1) for _ in range(self.context_dim)]
            # indicator for absorbing state
            state = torch.Tensor(np.append(np.append(current_observation, context), 0))
            done = False
            timestep = 0

            # Episode reward is used only as a metric for performance
            episode_reward = 0
            while not done and timestep <= self.trajectory_length - 2:
                ''' Parallelize here TODO
                 initialize n
                 states = []
                 actions =[]
                 for i in range(n):
                    state = env.reset()
                    states.append(state)
                    action = self.off_policy_algorithm.get_action(state)
                    actions.append(action)
                    
                 Scatter action
                 for each agent
                 observation, reward, done, _ = self.env.step(np.array(action), project=False)
                 Gather observation ,reward,done 
                 Append policy trajectory 
                 Wrap trajecory and add to buffer
               
                '''
                action = self.off_policy_algorithm.get_action(state).detach()
                observation, reward, done, _ = self.env.step(np.array(action), project=False)
                observation = get_policy_observation(observation)
                sample = dict(state=current_observation, action=action, reward=reward, is_absorbing=False,
                              next_state=observation, done=done)
                policy_trajectory.append(sample)

                current_observation = observation
                state = torch.Tensor(np.append(np.append(current_observation, context), 0))
                episode_reward += reward
                timestep += 1
            policy_trajectory[-1]["done"] = True
            non_encoded_absorbing_state = [0] * (current_observation.shape[0])
            policy_trajectory[-1]["next_state"] = np.array([non_encoded_absorbing_state]).T

            # Wrap policy trajectory with absorbing state and store in replay buffer
            policy_trajectory = {"trajectory": policy_trajectory, "context": context}
            self._wrap_trajectories([policy_trajectory])
            self.replay_buffer.add(policy_trajectory)

            self.writer.add_scalar("Episode reward", episode_reward, global_step=iteration)

            if episode_reward > self.max_reward:
                self.max_reward = episode_reward
                # save current best model
                print(f"\nNew best model with reward {self.max_reward}")
                self.save_model(all_nets_path=modeldir + logfile + "/", env_name="ProstheticsEnv", info="best",
                                attributes_path=attributesdir + logfile + "/")

            if iteration % save_iter == 0:
                print(f"\nSaving periodically - iteration {iteration}")
                self.save_model(all_nets_path=modeldir + logfile + "/", env_name="ProstheticsEnv", info="best",
                                attributes_path=attributesdir + logfile + "/")
                self.replay_buffer.save_buffer(path=bufferdir + logfile + "/", info="ProstheticsEnv")

            self.current_iteration += 1

            if len(self.replay_buffer) < self.batch_size:
                continue

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

                log_D_expert = []
                log_D_replay = []
                log_one_minus_D_replay = []
                encoder_kl_divergence = 0
                num_non_absorbing_states = 0
                discriminator_gradient_penalty = 0
                encoder_prior = Normal(0, 1)

                expert_discriminator_outputs = []
                replay_discriminator_outputs = []

                for i in range(len(expert_batch)):

                    # Encode the states
                    if not expert_batch[i]["is_absorbing"]:
                        expert_batch[i]["encoded_state"], expert_mean, expert_std, _, _ = self.encoder.sample(
                            torch.Tensor(expert_batch[i]["state"][:, 0]), info=True)

                        """Calculate KL divergence for encoder
                        Sum along individual dimensions gives KL div for multivariate since sigma is diagonal matrix"""
                        encoder_expert_distribution = Normal(expert_mean, expert_std)
                        expert_kl_divergence = torch.sum(kl_divergence(encoder_expert_distribution, encoder_prior))
                        encoder_kl_divergence += expert_kl_divergence
                        num_non_absorbing_states += 1

                    if not replay_batch[i]["is_absorbing"]:
                        replay_batch[i]["encoded_state"], replay_mean, replay_std, _, _ = self.encoder.sample(
                            torch.Tensor(replay_batch[i]["state"][:, 0]), info=True)
                        encoder_replay_distribution = Normal(replay_mean, replay_std)
                        replay_kl_divergence = torch.sum(kl_divergence(encoder_replay_distribution, encoder_prior))
                        encoder_kl_divergence += replay_kl_divergence
                        num_non_absorbing_states += 1

                    """Prepare observation for discriminator. 
                    Contains encoded_state, context, phase and absorbing indicator"""
                    expert_discriminator_input = self._prepare_discriminator_input(transition=expert_batch[i],
                                                                                   context=context, phase=phase)
                    replay_discriminator_input = self._prepare_discriminator_input(transition=replay_batch[i],
                                                                                   context=context, phase=phase)

                    # discriminator forward
                    expert_discriminator_outputs.append(self.discriminator(expert_discriminator_input))
                    replay_discriminator_outputs.append(self.discriminator(replay_discriminator_input))

                    # Calculate gradient penalties for discriminator
                    expert_gradients = autograd.grad(outputs=expert_discriminator_outputs[i],
                                                     inputs=expert_discriminator_input["input"],
                                                     create_graph=True, retain_graph=True, only_inputs=True)[0]

                    expert_gradient_penalty = (expert_gradients.norm() - 1) ** 2

                    replay_gradients = autograd.grad(outputs=replay_discriminator_outputs[i],
                                                     inputs=replay_discriminator_input["input"],
                                                     create_graph=True, retain_graph=True, only_inputs=True)[0]

                    replay_gradient_penalty = (replay_gradients.norm() - 1) ** 2

                    discriminator_gradient_penalty += expert_gradient_penalty + replay_gradient_penalty

                    # discriminator D is trained to output 0 for policy and 1 for expert
                    log_D_expert.append(torch.log(expert_discriminator_outputs[i]))
                    log_D_replay.append(torch.log(replay_discriminator_outputs[i]))
                    log_one_minus_D_replay.append(torch.log(1 - replay_discriminator_outputs[i]))

                log_D_expert_sum = torch.stack(log_D_expert).sum()
                log_one_minus_D_replay_sum = torch.stack(log_one_minus_D_replay).sum()

                if num_non_absorbing_states:
                    # normalise encoder kl divergence
                    encoder_kl_divergence /= num_non_absorbing_states

                expert_discriminator_outputs = torch.Tensor(expert_discriminator_outputs)
                replay_discriminator_outputs = torch.Tensor(replay_discriminator_outputs)

                self.writer.add_scalar("Encoder KL divergence", encoder_kl_divergence,
                                       global_step=iteration * self.trajectory_length + timestep)
                self.writer.add_scalar("Normalised discriminator gradient penalty",
                                       discriminator_gradient_penalty / (2 * self.trajectory_length),
                                       global_step=iteration * self.trajectory_length + timestep)

                self.writer.add_scalar("Discriminator output for expert (mean for current timestep)",
                                       expert_discriminator_outputs.mean(),
                                       global_step=iteration * self.trajectory_length + timestep)
                self.writer.add_scalar("Discriminator output for generator policy (mean for current timestep)",
                                       replay_discriminator_outputs.mean(),
                                       global_step=iteration * self.trajectory_length + timestep)

                # Calculate losses for encoder and discriminator
                if num_non_absorbing_states:
                    encoder_loss = -log_D_expert_sum - log_one_minus_D_replay_sum + self.beta * (
                            encoder_kl_divergence - self.information_constraint)
                    discriminator_loss = encoder_loss + self.gp_lambda * discriminator_gradient_penalty
                else:
                    discriminator_loss = -log_D_expert_sum - log_one_minus_D_replay_sum +\
                                         self.gp_lambda * discriminator_gradient_penalty

                # Update encoder
                if num_non_absorbing_states:
                    self.encoder_optimizer.zero_grad()
                    encoder_loss.backward(retain_graph=True)
                    self.encoder_optimizer.step()

                    # Update beta
                    self.beta = max(0, self.beta + self.beta_lr * (
                            encoder_kl_divergence - self.information_constraint)).detach()

                    self.writer.add_scalar("Encoder loss", encoder_loss,
                                           global_step=iteration * self.trajectory_length + timestep)

                # Update discriminator
                None_grad(self.discriminator_optimizer)
                discriminator_loss.backward()
                self.discriminator_optimizer.step()

                self.writer.add_scalar("Discriminator loss", discriminator_loss,
                                       global_step=iteration * self.trajectory_length + timestep)
                self.writer.add_scalar("Beta", self.beta,
                                       global_step=iteration * self.trajectory_length + timestep)

                """Prepare batch for off policy update
                A batch is a dictionary of lists containing state, action, next_state, reward and done 
                """
                batch = {"state": [], "action": [], "reward": [], "done": [], "next_state": []}

                for i in range(len(expert_batch)):
                    if not replay_batch[i]["is_absorbing"]:
                        batch["state"].append(
                            torch.cat([torch.Tensor(replay_batch[i]["state"][:, 0]), torch.Tensor(context),
                                       torch.Tensor([0])]))
                        batch["action"].append(torch.Tensor(replay_batch[i]["action"]))
                        batch["done"].append(torch.Tensor([replay_batch[i]["done"]]))
                        batch["next_state"].append(
                            torch.cat([torch.Tensor(replay_batch[i]["next_state"][:, 0]), torch.Tensor(context),
                                       torch.Tensor([int(replay_batch[i]["done"])])]))

                        reward = log_D_replay[i] - log_one_minus_D_replay[i]
                        if self.reward_clip:
                            reward = torch.clamp(reward, min=-self.clip_val_reward, max=self.clip_val_reward)
                        batch["reward"].append(torch.Tensor([reward]))

                if len(batch["state"]):
                    self.policy_update_count += 1
                    self.off_policy_algorithm.policy_update(batch=batch, update_number=self.policy_update_count)

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

    def _prepare_discriminator_input(self, transition, phase, context):

        if transition["is_absorbing"]:
            return {"input": torch.Tensor(self.absorbing_state).requires_grad_(), "phase": phase}
        else:
            return {"input": torch.cat([transition["encoded_state"],
                                        torch.Tensor(context),
                                        torch.Tensor([0])]), "phase": phase}

    def save_model(self, env_name, attributes_path=None, all_nets_path=None, discriminator_path=None, encoder_path=None,
                   actor_path=None, critic_path=None, value_path=None, info="none"):
        self.off_policy_algorithm.save_model(env_name=env_name, all_nets_path=all_nets_path, actor_path=actor_path,
                                             critic_path=critic_path, value_path=value_path, info=info)
        if all_nets_path is not None:
            discriminator_path = all_nets_path
            encoder_path = all_nets_path

        if discriminator_path is None:
            discriminator_path = f'model/{env_name}/'
        os.makedirs(discriminator_path, exist_ok=True)

        if encoder_path is None:
            encoder_path = f'model/{env_name}/'
        os.makedirs(encoder_path, exist_ok=True)

        if attributes_path is None:
            attributes_path = f"attributes/{env_name}"
        os.makedirs(attributes_path, exist_ok=True)

        print_heading("Saving discriminator and encoder network parameters")
        torch.save(self.discriminator.state_dict(), discriminator_path + f"actor_{info}.pt")
        torch.save(self.encoder.state_dict(), encoder_path + f"value_{info}.pt")

        with open(attributes_path+"attributes.pkl","wb") as f:
            pickle.dump({"current_iteration": self.current_iteration, "beta": self.beta,
                         "policy_update_count": self.policy_update_count, "max_reward": self.max_reward}, f)
        heading_decorator(bottom=True, print_req=True)

    def load_model(self, attributes_path, discriminator_path=None, encoder_path=None, actor_path=None, critic_path=None,
                   value_path=None):
        self.off_policy_algorithm.load_model(actor_path=actor_path, critic_path=critic_path, value_path=value_path)
        print_heading("Loading models from paths: \n discriminator:{} \n encoder:{} \n attributes:{}"
                      .format(discriminator_path, encoder_path, attributes_path))
        if discriminator_path is not None:
            self.discriminator.load_state_dict(torch.load(discriminator_path))
        if encoder_path is not None:
            self.encoder.load_state_dict(torch.load(encoder_path))

        with open(attributes_path,"rb") as f:
            attributes = pickle.load(f)

        self.current_iteration = attributes["current_iteration"]
        self.beta = attributes["beta"]
        self.policy_update_count = attributes["policy_update_count"]
        self.max_reward = attributes["max_reward"]

        print('loading done')
