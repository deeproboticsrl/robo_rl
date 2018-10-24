import os

import numpy as np
import torch
import torch.nn.functional as torchfunc
from osim.env import ProstheticsEnv
from pros_ai import get_policy_observation, get_expert_observation
from robo_rl.common import LinearPFDiscriminator, Buffer, LinearGaussianNetwork, no_activation, TrajectoryBuffer
from robo_rl.obsvail import ObsVAIL
from robo_rl.obsvail import get_obsvail_parser, get_logfile_name
from robo_rl.sac import SAC, SigmoidSquasher
from tensorboardX import SummaryWriter
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
# TODO make it automatic
context_dim = 2
# According to VAIL
sac_hidden_dim = [1024, 512]

logdir = "./tensorboard_log/"
# logdir += "dummy"
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
          td3_update_interval=args.td3_update_interval, squasher=squasher, weight_decay=args.weight_decay,
          grad_clip=args.grad_clip, loss_clip=args.loss_clip, clip_val_grad=args.clip_val_grad,
          deterministic=args.deterministic, clip_val_loss=args.clip_val_loss, log_std_min=args.log_std_min,
          log_std_max=args.log_std_max)

buffer = Buffer(capacity=args.replay_buffer_capacity)
expert_buffer = TrajectoryBuffer(capacity=args.expert_buffer_capacity)

expert_file_path = "./experts/sampled_experts.obs"

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
encoder = LinearGaussianNetwork(layers_size=encoder_layer_sizes, final_layer_function=no_activation,
                                activation_function=torchfunc.relu, is_layer_norm=False)

# TODO get from argparse lr decay and steps
obsvail = ObsVAIL(env=env, expert_file_path=expert_file_path, discriminator=discriminator, off_policy_algorithm=sac,
                  encoder=encoder, replay_buffer_capacity=args.replay_buffer_capacity,context_dim=context_dim,
                  absorbing_state_dim=discriminator_input_dim, writer=writer, beta_lr=args.beta_lr,
                  discriminator_lr=args.discriminator_lr, encoder_lr=args.encoder_lr, beta_init=args.beta_init,
                  learning_rate_decay=22, learning_rate_decay_training_steps=22, optimizer=optimizer,
                  weight_decay=args.weight_decay, grad_clip=args.grad_clip, loss_clip=args.loss_clip,
                  clip_val_grad=args.clip_val_grad, clip_val_loss=args.clip_val_loss)

obsvail.train(num_iterations=args.num_iterations, save_iter=args.save_iter)

# TODO Gradient clipping in actor net

# TODO For SAC use reparam trick with normalising flow(??)

# TODO regularisation in form of gradient penalties for stable learning. makes GAN stable. Refer paper
# TODO Should we use simple weight regularisation then?
