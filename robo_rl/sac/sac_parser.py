import argparse

parser = argparse.ArgumentParser(description='Soft Actor Critic')
parser.add_argument('--env_seed', type=int, default=0, help="environment seed")
parser.add_argument('--soft_update_tau', type=float, default=0.005, help="target smoothing coefficient tau")
parser.add_argument('--policy_lr', type=float, default=0.0003, help="policy learning rate")
parser.add_argument('--value_lr', type=float, default=0.0003, help="value net learning rate")
parser.add_argument('--critic_lr', type=float, default=0.0003, help="q value net learning rate")

parser.add_argument('--policy_weight_decay', type=float, default=0,
                    help="L2 regularisation constant for policy weights")
parser.add_argument('--value_weight_decay', type=float, default=0,
                    help="L2 regularisation constant for value weights")
parser.add_argument('--critic_weight_decay', type=float, default=0,
                    help="L2 regularisation constant for critic weights")

parser.add_argument('--discount_factor', type=float, default=0.99, help='discount factor gamma')
parser.add_argument('--scale_reward', type=float, default=10000,
                    help="reward scaling humannoid_v1=20, humnanoid_rllab=10, other mujoco=5")
parser.add_argument('--reparam', type=bool, default=True, help="True if reparameterization trick is applied")
parser.add_argument('--rewarding', type=bool, default=False, help="Hindsight reward for each transition")
parser.add_argument('--unbiased', type=bool, default=True, help="Apply importance sampling to hindsight transitions")
parser.add_argument('--goal_obs', type=bool, default=True, help="Whether to use complete observations or only achieved "
                                                                "goal part as state")
parser.add_argument('--positive_reward', type=bool, default=True,
                    help="Add 1 to the binary rewards {-1,0} to make them {0,1}")

parser.add_argument('--deterministic', type=bool, default=False)
parser.add_argument('--grad_clip', type=bool, default=False, help="Whether to clip gradients in each update")
parser.add_argument('--loss_clip', type=bool, default=False, help="Whether to clip losses in each update")
parser.add_argument('--clip_val_loss', type=float, default=1000, help="Max value(absolute) for losses when clipping")
parser.add_argument('--clip_val_grad', type=float, default=0.01, help="Max value(absolute) for gradients when clipping")
parser.add_argument('--log_std_min', type=float, default=-20)
parser.add_argument('--log_std_max', type=float, default=2)

parser.add_argument('--target_update_interval', type=int, default=1,
                    help="used in case of hard update for the value function")
parser.add_argument('--td3_update_interval', type=int, default=100,
                    help="used in case of delayed update for policy")

parser.add_argument('--hidden_dim', type=int, default=256, help='no of hidden units')
parser.add_argument('--num_layers', type=int, default=2, help='no of hidden layers')
parser.add_argument('--buffer_capacity', type=int, default=100000, help='buffer capacity')
parser.add_argument('--sample_batch_size', type=int, default=256, help='number of samples from replay buffer')
parser.add_argument('--max_time_steps', type=int, default=10000, help='max number of env timesteps per episodes')
parser.add_argument('--num_episodes', type=int, default=100000, help='number of episodes')
parser.add_argument('--updates_per_step', type=int, default=1, help='updates per step')
parser.add_argument('--save_iter', type=int, default=100, help='save model and buffer '
                                                               'after certain number of iteration')
parser.add_argument('--test_interval', type=int, default=25, help="Number of episodes after which to test")
parser.add_argument('--num_tests', type=int, default=50, help="Number of tests for evaluation")


def get_sac_parser():
    return parser


def get_logfile_name(args):
    logfile = ""
    # if args.unbiased:
    #     logfile += "unbiased_her"
    # else:
    #     logfile += "biased_her"
    #
    # if args.rewarding:
    #     logfile += "_rewarding"
    # else:
    #     logfile += "_unrewarding"

    if args.reparam:
        logfile += "_reparam"
    else:
        logfile += "_no_reparam"

    # if args.positive_reward:
    #     logfile += "_positive_reward"

    logfile += f"_reward_scale={args.scale_reward}_tau={args.soft_update_tau}"
    logfile += f"_samples={args.sample_batch_size}_discount_factor={args.discount_factor}"
    logfile += f"_td3={args.td3_update_interval}"
    logfile += f"_updates={args.updates_per_step}_num_episodes={args.num_episodes}"
    logfile += f"_log_std_min={args.log_std_min}_max={args.log_std_max}_seed={args.env_seed}"
    logfile += f"_hidden_layers={args.num_layers}_nodes_{args.hidden_dim}"

    # if args.goal_obs:
    #     logfile += "_GOALIFIED_states"
    if args.grad_clip:
        logfile += f"_grad_clip_{args.clip_val_grad}"
    if args.loss_clip:
        logfile += f"_loss_clip_{args.clip_val_loss}"

    return logfile
