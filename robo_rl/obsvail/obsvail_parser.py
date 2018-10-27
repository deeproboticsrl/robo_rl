import argparse

parser = argparse.ArgumentParser(description='GAIL using observations only')
parser.add_argument('--env_seed', type=int, default=0, help="environment seed")
parser.add_argument('--grad_clip', type=bool, default=False, help="Whether to clip gradients in each update")
parser.add_argument('--loss_clip', type=bool, default=False, help="Whether to clip losses in each update")
parser.add_argument('--reward_clip', type=bool, default=False, help="Whether to clip reward in each update")
parser.add_argument('--clip_val_loss', type=float, default=1000, help="Max value(absolute) for losses when clipping")
parser.add_argument('--clip_val_grad', type=float, default=40, help="Max value(absolute) for gradients when clipping")
parser.add_argument('--clip_val_reward', type=float, default=1, help="Max value(absolute) for reward when clipping")

parser.add_argument('--num_workers', type=int, default=15, help='Number of workers for each evaluation')

parser.add_argument('--save_iter', type=int, default=20, help='save model and buffer '
                                                              'after certain number of iteration')
parser.add_argument('--max_time_steps', type=int, default=100, help='max number of env timesteps per episode')
parser.add_argument('--num_iterations', type=int, default=1000000, help='number of iterations of the main loop')

# Soft Actor Critic hyper-parameters
parser.add_argument('--policy_lr', type=float, default=0.0003, help="learning rate for policy")
parser.add_argument('--value_lr', type=float, default=0.0003, help="learning rate for state value function")
parser.add_argument('--critic_lr', type=float, default=0.0003, help="learning rate for state-action value function")

parser.add_argument('--soft_update_tau', type=float, default=0.005, help="target smoothening coefficient tau")
parser.add_argument('--discount_factor', type=float, default=0.99, help='discount factor gamma')
parser.add_argument('--scale_reward', type=float, default=1000,
                    help="reward scaling humannoid_v1=20, humnanoid_rllab=10, other mujoco=5")
parser.add_argument('--reparam', type=bool, default=True, help="True if reparameterization trick is applied")

parser.add_argument('--deterministic', type=bool, default=False)
parser.add_argument('--log_std_min', type=float, default=-20)
parser.add_argument('--log_std_max', type=float, default=2)

parser.add_argument('--target_update_interval', type=int, default=1,
                    help="used in case of hard update for the value function")
parser.add_argument('--td3_update_interval', type=int, default=1,
                    help="used in case of delayed update for policy")

parser.add_argument('--replay_buffer_capacity', type=int, default=10000, help='replay buffer capacity')
parser.add_argument('--expert_buffer_capacity', type=int, default=120, help='expert buffer capacity')
parser.add_argument('--batch_size', type=int, default=32, help='number of samples from buffer used for 1 update')

parser.add_argument('--num_networks_discriminator', type=int, default=10,
                    help='number of intervals in Phase Functional discriminator')
parser.add_argument('--information_constraint', type=float, default=0.2, help='value of bottleneck constraint')
parser.add_argument('--gp_lambda', type=float, default=10, help='Coefficient for gradient penalties')

parser.add_argument('--beta_lr', type=float, default=0.001, help='stepsize for updating beta in vail')
parser.add_argument('--discriminator_lr', type=float, default=0.0003, help="learning rate for discriminator")
parser.add_argument('--encoder_lr', type=float, default=0.0003, help="learning rate for encoder")

parser.add_argument('--beta_init', type=float, default=0.1, help='initial value for beta')

parser.add_argument('--encoder_weight_decay', type=float, default=0.1,
                    help="L2 regularisation constant for encoder weights")
parser.add_argument('--discriminator_weight_decay', type=float, default=0,
                    help="L2 regularisation constant for discriminator weights")
parser.add_argument('--policy_weight_decay', type=float, default=0,
                    help="L2 regularisation constant for policy weights")
parser.add_argument('--value_weight_decay', type=float, default=0,
                    help="L2 regularisation constant for value weights")
parser.add_argument('--critic_weight_decay', type=float, default=0,
                    help="L2 regularisation constant for critic weights")

parser.add_argument('--use_rl_reward', type=bool, default=False,
                    help="Whether to use rl reward in addition to discriminator's reward")


def get_obsvail_parser():
    return parser


def get_logfile_name(args):
    logfile = ""

    if args.reparam:
        logfile += "_reparam"
    else:
        logfile += "_no_reparam"

    logfile += f"_reward_scale={args.scale_reward}_tau={args.soft_update_tau}_beta_init_{args.beta_init}"
    logfile += f"_samples={args.batch_size}_discount_factor={args.discount_factor}"
    logfile += f"_encoder_weight_decay={args.encoder_weight_decay}_num_iterations={args.num_iterations}"
    logfile += f"_LR_policy_{args.policy_lr}_critic_{args.critic_lr}_value_{args.value_lr}"
    logfile += f"_discriminator_{args.discriminator_lr}_encoder_{args.encoder_lr}_beta_{args.beta_lr}"

    if args.reward_clip:
        logfile += f"_reward_clip"

    return logfile
