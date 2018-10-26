import os

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as torchfunc
from osim.env import ProstheticsEnv
from pros_ai import get_policy_observation, get_expert_observation
from robo_rl.common import LinearPFDiscriminator, no_activation, LinearGaussianEncoder, print_heading, None_grad
from robo_rl.obsvail import ObsVAIL
from robo_rl.obsvail import get_obsvail_parser, get_logfile_name
from robo_rl.sac import SAC, SigmoidSquasher
from tensorboardX import SummaryWriter
from torch.distributions import Normal, kl_divergence
from torch.optim import Adam

optimizer = Adam

parser = get_obsvail_parser()
args = parser.parse_args()
env = ProstheticsEnv(visualize=False)

# seeding
env.seed(args.env_seed)
torch.manual_seed(args.env_seed)
np.random.seed(args.env_seed)

observation = env.reset(project=False)
action_dim = env.action_space.shape[0]

"""This is done to allow having different observations for policy and discriminator
"""
policy_state_dim = get_policy_observation(observation).shape[0]
context_dim = 2

# According to VAIL
sac_hidden_dim = [1024, 512]

# logdir = "./tensorboard_log/"
logdir = "dummy"
modeldir = f"./model/ProstheticsEnv/"
bufferdir = f"./buffer/ProstheticsEnv/"

logfile = get_logfile_name(args)

os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(log_dir=logdir)

squasher = SigmoidSquasher()

sac = SAC(action_dim=action_dim, state_dim=policy_state_dim + context_dim, hidden_dim=sac_hidden_dim,
          discount_factor=args.discount_factor, optimizer=optimizer, policy_lr=args.policy_lr, critic_lr=args.critic_lr,
          value_lr=args.value_lr, writer=writer, scale_reward=args.scale_reward, reparam=args.reparam,
          target_update_interval=args.target_update_interval, soft_update_tau=args.soft_update_tau,
          td3_update_interval=args.td3_update_interval, squasher=squasher, policy_weight_decay=args.policy_weight_decay,
          critic_weight_decay=args.critic_weight_decay, value_weight_decay=args.value_weight_decay,
          grad_clip=args.grad_clip, loss_clip=args.loss_clip, clip_val_grad=args.clip_val_grad,
          deterministic=args.deterministic, clip_val_loss=args.clip_val_loss, log_std_min=args.log_std_min,
          log_std_max=args.log_std_max)

expert_file_path = "../experts/sampled_experts.obs"

expert_state_dim = get_expert_observation(observation).shape[0]
latent_z_dim = int(expert_state_dim / 3)

"""Add 1 dimension for absorbing state
This state isn't needed in the policy
"""
discriminator_input_dim = latent_z_dim + context_dim + 1
discriminator_hidden_dim = [512]
discriminator = LinearPFDiscriminator(input_dim=discriminator_input_dim, hidden_dim=discriminator_hidden_dim,
                                      num_networks=args.num_networks_discriminator)

encoder_layer_sizes = [expert_state_dim]
encoder_hidden_dim = [int(expert_state_dim / 1.57), int(expert_state_dim / 2), int(expert_state_dim / 2.35)]
encoder_layer_sizes.extend(encoder_hidden_dim)
encoder_layer_sizes.append(latent_z_dim)
encoder = LinearGaussianEncoder(layers_size=encoder_layer_sizes, final_layer_function=no_activation,
                                activation_function=torchfunc.relu)

# TODO get from argparse lr decay and steps
obsvail = ObsVAIL(env=env, expert_file_path=expert_file_path, discriminator=discriminator, off_policy_algorithm=sac,
                  encoder=encoder, replay_buffer_capacity=args.replay_buffer_capacity, context_dim=context_dim,
                  absorbing_state_dim=discriminator_input_dim, writer=writer, beta_lr=args.beta_lr,
                  discriminator_lr=args.discriminator_lr, encoder_lr=args.encoder_lr, beta_init=args.beta_init,
                  learning_rate_decay=22, learning_rate_decay_training_steps=22, optimizer=optimizer,
                  discriminator_weight_decay=args.discriminator_weight_decay,
                  encoder_weight_decay=args.encoder_weight_decay,
                  grad_clip=args.grad_clip, loss_clip=args.loss_clip,
                  clip_val_grad=args.clip_val_grad, clip_val_loss=args.clip_val_loss, batch_size=args.batch_size)

# Test expert buffer. Size and wrapping
print_heading("Trajectory Length")
print(obsvail.trajectory_length)

print_heading("Expert buffer length")
print(len(obsvail.expert_buffer))

expert_trajectory = obsvail.expert_buffer.sample(batch_size=1)["trajectory"]
print_heading("Sampled expert trajectory details")
print("Trajectory Length ".ljust(50), len(expert_trajectory))
print("Absorbing indicator for 1st state".ljust(50), expert_trajectory[0]["is_absorbing"])
print("Absorbing indicator for last state".ljust(50), expert_trajectory[-1]["is_absorbing"])
print("Absorbing indicator for 2nd last state".ljust(50), expert_trajectory[-2]["is_absorbing"])

print_heading("SAC episode sampling")

policy_trajectory = []
observation = get_policy_observation(obsvail.env.reset(project=False))
# Sample random context for the trajectory
context = [np.random.randint(0, 1) for _ in range(obsvail.context_dim)]
print("Random trajectory context ".ljust(50), context)

state = torch.Tensor(np.append(observation, context))
print("Initial state".ljust(50), state)

done = False
timestep = 0
episode_reward = 0

while not done and timestep <= 2:
    # while not done and timestep <= obsvail.trajectory_length - 2:
    # used only as a metric for performance
    action = obsvail.off_policy_algorithm.get_action(state).detach()
    observation, reward, done, _ = obsvail.env.step(np.array(action), project=False)
    observation = get_policy_observation(observation)
    sample = dict(state=observation, action=action, reward=reward, is_absorbing=False)
    policy_trajectory.append(sample)

    state = torch.Tensor(np.append(observation, context))
    episode_reward += reward
    timestep += 1

# Wrap policy trajectory with absorbing state and store in replay buffer
policy_trajectory = {"trajectory": policy_trajectory, "context": context}
obsvail._wrap_trajectories([policy_trajectory])
obsvail.replay_buffer.add(policy_trajectory)

replay_buffer_sample = obsvail.replay_buffer.sample(batch_size=1)
sampled_policy_trajectory_context = replay_buffer_sample["context"]
sampled_policy_trajectory = replay_buffer_sample["trajectory"]
print_heading("Sampled policy trajectory details")
print("Trajectory Length ".ljust(50), len(sampled_policy_trajectory))
print("Absorbing indicator for 1st state".ljust(50), sampled_policy_trajectory[0]["is_absorbing"])
print("Absorbing indicator for last state".ljust(50), sampled_policy_trajectory[-1]["is_absorbing"])
print("Trajectory context".ljust(50), sampled_policy_trajectory_context)
print("Episode Reward".ljust(50), episode_reward)

""" For each timestep, sample a mini-batch from both expert and replay buffer.
Use it to update discriminator, encoder, policy, and beta
"""
timestep = 0
phase = min(1.0, timestep / (obsvail.trajectory_length - 2))

""" Expert observations are dictionary containing state, context and absorbing state indicator
Replay observations additionally have action and RL reward from the environment
"""
expert_batch = obsvail.expert_buffer.sample_timestep(batch_size=1, timestep=timestep)
replay_batch = obsvail.replay_buffer.sample_timestep(batch_size=1, timestep=timestep)

# Encode the states
for i in range(len(expert_batch)):
    if not expert_batch[i]["is_absorbing"]:
        expert_batch[i]["encoded_state"] = obsvail.encoder.sample(torch.Tensor(expert_batch[i]["state"][:, 0]))
    if not replay_batch[i]["is_absorbing"]:
        replay_batch[i]["encoded_state"] = obsvail.encoder.sample(torch.Tensor(replay_batch[i]["state"][:, 0]))

print_heading("Encoded states")
print("Expert encoded state".ljust(30), expert_batch[0]["encoded_state"])
print("Replay encoded state".ljust(30), replay_batch[0]["encoded_state"])
print("Encoded state dimensions expert".ljust(30), expert_batch[0]["encoded_state"].shape)
print("Encoded state dimensions replay".ljust(30), replay_batch[0]["encoded_state"].shape)

# Test kl divergence backprop update on encoder
print_heading("KL divergence update check")

z, mean, std, _, _ = obsvail.encoder.sample(torch.Tensor(expert_batch[0]["state"][:, 0]), info=True)
print("mean before update".ljust(30), mean)
print("std before update".ljust(30), std)

num_updates = 5
encoder_prior = Normal(0, 1)
for _ in range(num_updates):
    encoder_distribution = Normal(mean, std)
    # Sum along individual dimensions gives KL div for multivariate since sigma is diagonal matrix
    encoder_kl_divergence = torch.sum(kl_divergence(encoder_distribution, encoder_prior))

    obsvail.encoder_optimizer.zero_grad()
    encoder_kl_divergence.backward()
    # print(encoder.linear_layers[0].weight.grad)
    obsvail.encoder_optimizer.step()

    z, mean, std, _, _ = obsvail.encoder.sample(torch.Tensor(expert_batch[0]["state"][:, 0]), info=True)

print(f"mean after {num_updates} updates".ljust(30), mean)
print(f"std after {num_updates} updates".ljust(30), std)

# Calculate discriminator output
for i in range(len(expert_batch)):
    expert_discriminator_input = obsvail._prepare_discriminator_input(transition=expert_batch[i],
                                                                      context=context, phase=phase)
    replay_discriminator_input = obsvail._prepare_discriminator_input(transition=replay_batch[i],
                                                                      context=context, phase=phase)

    # discriminator forward
    expert_batch[i]["discriminator_output"] = obsvail.discriminator(expert_discriminator_input)
    replay_batch[i]["discriminator_output"] = obsvail.discriminator(replay_discriminator_input)

    print_heading("Discriminator input and output")
    print("Expert discriminator input".ljust(30), expert_discriminator_input)
    print("Replay discriminator input".ljust(30), replay_discriminator_input)
    print("Expert discriminator output".ljust(30), expert_batch[i]["discriminator_output"])
    print("Replay discriminator output".ljust(30), replay_batch[i]["discriminator_output"])

    gradients = autograd.grad(outputs=expert_batch[i]["discriminator_output"],
                              inputs=expert_discriminator_input["input"],
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = (gradients.norm() - 1)**2
    print(gradients)
    print(gradient_penalty)
    None_grad(obsvail.discriminator_optimizer)
    gradient_penalty.backward(retain_graph=True)
    obsvail.discriminator_optimizer.step()

    for _ in range(num_updates):
        # discriminator forward
        expert_batch[i]["discriminator_output"] = obsvail.discriminator(expert_discriminator_input)
        replay_batch[i]["discriminator_output"] = obsvail.discriminator(replay_discriminator_input)
        gradients = autograd.grad(outputs=expert_batch[i]["discriminator_output"],
                                  inputs=expert_discriminator_input["input"],
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = (gradients.norm() - 1) ** 2
        None_grad(obsvail.discriminator_optimizer)
        gradient_penalty.backward(retain_graph=True)
        obsvail.discriminator_optimizer.step()

    print_heading(f"Discriminator input and output after {num_updates} gradient penalty updates")
    print("Expert discriminator input".ljust(30), expert_discriminator_input)
    print("Replay discriminator input".ljust(30), replay_discriminator_input)
    print("Expert discriminator output".ljust(30), expert_batch[i]["discriminator_output"])
    print("Replay discriminator output".ljust(30), replay_batch[i]["discriminator_output"])

    print(gradients)
    print(gradient_penalty)

