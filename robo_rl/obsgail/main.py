import os

import numpy as np
import torch
from osim.env import ProstheticsEnv
from robo_rl.common import LinearDiscriminator, Buffer
from robo_rl.obsgail import ExpertBuffer, ObsGAIL
from robo_rl.sac import SAC, SigmoidSquasher
from robo_rl.obsgail import get_obsgail_parser, get_logfile_name
from tensorboardX import SummaryWriter
from torch.optim import Adam

parser = get_obsgail_parser()
args = parser.parse_args()
env = ProstheticsEnv()

env.seed(args.env_seed)
torch.manual_seed(args.env_seed)
np.random.seed(args.env_seed)

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
# According to VAIL
sac_hidden_dim = [1024, 512]

logdir = "./tensorboard_log/"
# logdir += "dummy"
modeldir = f"./model/{args.env_name}/"
bufferdir = f"./buffer/{args.env_name}"

logfile = get_logfile_name(args)

os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(log_dir=logdir)

squasher = SigmoidSquasher()

sac = SAC(action_dim=action_dim, state_dim=state_dim, hidden_dim=sac_hidden_dim,
          discount_factor=args.discount_factor, optimizer=Adam, policy_lr=args.policy_lr, critic_lr=args.critic_lr,
          value_lr=args.value_lr, writer=writer, scale_reward=args.scale_reward, reparam=args.reparam,
          target_update_interval=args.target_update_interval, soft_update_tau=args.soft_update_tau,
          td3_update_interval=args.td3_update_interval, squasher=squasher, weight_decay=args.weight_decay,
          grad_clip=args.grad_clip, loss_clip=args.loss_clip, clip_val_grad=args.clip_val_grad,
          deterministic=args.deterministic, clip_val_loss=args.clip_val_loss, log_std_min=args.log_std_min,
          log_std_max=args.log_std_max)

buffer = Buffer(capacity=args.replay_buffer_capacity)
expert_buffer = ExpertBuffer(capacity=args.expert_buffer_capacity)


# Fill expert buffer
expert_file_path = "./experts/sampled_experts.obs"
expert_buffer.add_from_file(expert_file_path=expert_file_path)


latent_z_dim = int(state_dim/3)
discriminator_input_dim = latent_z_dim
discriminator_hidden_dim = [512]

discriminator = LinearDiscriminator(input_dim=discriminator_input_dim, hidden_dim=discriminator_hidden_dim)

obsgail = ObsGAIL(env=env,expert_buffer=expert_buffer, discriminator=discriminator, off_policy_algo=sac)

# TODO get from argparse
# obsgail.train(num_iterations=,learning_rate=,learning_rate_decay=,learning_rate_decay_training_steps=)

# TODO Gradient clipping in actor net

# TODO For SAC use reparam trick with normalising flow(??)

# TODO regularisation in form of gradient penalties for stable learning. makes GAN stable. Refer paper
# TODO Should we use simple weight regularisation then?
