import argparse

parser = argparse.ArgumentParser(description='GAIL using observations only')
parser.add_argument('--env_seed', type=int, default=0, help="environment seed")
parser.add_argument('--weight_decay', type=float, default=0, help="regularisation constant for network weights")
parser.add_argument('--grad_clip', type=bool, default=False, help="Whether to clip gradients in each update")
parser.add_argument('--loss_clip', type=bool, default=False, help="Whether to clip losses in each update")
parser.add_argument('--clip_val_loss', type=float, default=1000, help="Max value(absolute) for losses when clipping")
parser.add_argument('--clip_val_grad', type=float, default=0.01, help="Max value(absolute) for gradients when clipping")

parser.add_argument('--save_iter', type=int, default=20, help='save model and buffer '
                                                              'after certain number of iteration')
parser.add_argument('--test_interval', type=int, default=20, help="Number of episodes after which to test")
parser.add_argument('--num_tests', type=int, default=50, help="Number of tests for evaluation")
parser.add_argument('--max_time_steps', type=int, default=100, help='max number of env timesteps per episodes')
parser.add_argument('--num_iterations', type=int, default=1000, help='number of iterations of the main loop')

# Soft Actor Critic hyper-parameters
parser.add_argument('--policy_lr', type=float, default=0.0003, help="learning rate for policy")
parser.add_argument('--value_lr', type=float, default=0.0003, help="learning rate for state value function")
parser.add_argument('--critic_lr', type=float, default=0.0003, help="learning rate for state-action value function")

parser.add_argument('--soft_update_tau', type=float, default=0.005, help="target smoothening coefficient tau")
parser.add_argument('--discount_factor', type=float, default=0.99, help='discount factor gamma')
parser.add_argument('--scale_reward', type=float, default=10000,
                    help="reward scaling humannoid_v1=20, humnanoid_rllab=10, other mujoco=5")
parser.add_argument('--reparam', type=bool, default=True, help="True if reparameterization trick is applied")

parser.add_argument('--deterministic', type=bool, default=False)
parser.add_argument('--log_std_min', type=float, default=-20)
parser.add_argument('--log_std_max', type=float, default=2)

parser.add_argument('--target_update_interval', type=int, default=1,
                    help="used in case of hard update for the value function")
parser.add_argument('--td3_update_interval', type=int, default=1,
                    help="used in case of delayed update for policy")

parser.add_argument('--replay_buffer_capacity', type=int, default=1000000, help='buffer capacity')
parser.add_argument('--expert_buffer_capacity', type=int, default=15000, help='buffer capacity')
parser.add_argument('--sample_batch_size', type=int, default=32, help='number of samples from replay buffer')
parser.add_argument('--updates_per_step', type=int, default=1, help='updates per step')

parser.add_argument('--num_networks_discriminator', type=int, default=10,
                    help='number of intervals in Phase Functional discriminator')
parser.add_argument('--information_constraint', type=float, default=0.5, help='value of bottleneck constraint')

parser.add_argument('--beta_lr', type=float, default=0.00001, help='stepsize for updating beta in vail')
parser.add_argument('--discriminator_lr', type=float, default=0.0003, help="learning rate for discriminator")
parser.add_argument('--encoder_lr', type=float, default=0.0003, help="learning rate for encoder")

parser.add_argument('--beta_init', type=float, default=0.00001, help='initial value for beta')


def get_obsvail_parser():
    return parser


def get_logfile_name(args):
    logfile = ""

    if args.reparam:
        logfile += "_reparam"
    else:
        logfile += "_no_reparam"

    logfile += f"_reward_scale={args.scale_reward}_tau={args.soft_update_tau}"
    logfile += f"_samples={args.sample_batch_size}_discount_factor={args.discount_factor}"
    logfile += f"_td3={args.td3_update_interval}"
    logfile += f"_updates={args.updates_per_step}_num_iterations={args.num_iterations}"
    logfile += f"_log_std_min={args.log_std_min}_max={args.log_std_max}_seed={args.env_seed}"
    logfile += f"_LR_policy_{args.policy_lr}_critic_{args.critic_lr}_value_{args.value_lr}"

    if args.grad_clip:
        logfile += f"_grad_clip"
    if args.loss_clip:
        logfile += f"_loss_clip"

    return logfile
