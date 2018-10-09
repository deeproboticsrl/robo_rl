import argparse

parser = argparse.ArgumentParser(description='PyTorch on fire')
parser.add_argument('--env_seed', type=int, default=1105, help="environment seed")
parser.add_argument('--soft_update_tau', type=float, default=0.005, help="target smoothening coefficient tau")
parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
parser.add_argument('--weight_decay', type=float, default=0.001, help="regularisation constant for network weights")
parser.add_argument('--discount_factor', type=float, default=0.999, help='discount factor gamma')
parser.add_argument('--scale_reward', type=float, default=0.001,
                    help="reward scaling humannoid_v1=20, humnanoid_rllab=10, other mujoco=5")
parser.add_argument('--reparam', type=bool, default=True, help="True if reparameterization trick is applied")
parser.add_argument('--deterministic', type=bool, default=False)
parser.add_argument('--target_update_interval', type=int, default=1,
                    help="used in case of hard update with or without td3")
parser.add_argument('--td3_update_interval', type=int, default=20,
                    help="used in case of delayed update for policy")

parser.add_argument('--hidden_dim', type=int, default=256, help='no of hidden units ')
parser.add_argument('--buffer_capacity', type=int, default=1000000, help='buffer capacity')
parser.add_argument('--sample_batch_size', type=int, default=32, help='number of samples from replay buffer')
parser.add_argument('--max_time_steps', type=int, default=10000, help='max number of env timesteps per episodes')
parser.add_argument('--num_episodes', type=int, default=101, help='number of episodes')
parser.add_argument('--updates_per_step', type=int, default=100, help='updates per step')
parser.add_argument('--save_iter', type=int, default=20, help='save model and buffer '
                                                              'after certain number of iteration')


def get_sac_parser():
    return parser
