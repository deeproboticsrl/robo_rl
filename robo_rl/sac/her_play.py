import gym
import numpy as np
import torch
from robo_rl.sac import SAC, TanhSquasher
from robo_rl.sac import get_sac_parser
from torch.optim import Adam
from robo_rl.common.utils import gym_torchify

parser = get_sac_parser()
parser.add_argument('--env_name', default="FetchReach-v1", help="Should be GoalEnv")
args = parser.parse_args()

env = gym.make(args.env_name)

env.seed(args.env_seed)
torch.manual_seed(args.env_seed)
np.random.seed(args.env_seed)

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.spaces["observation"].shape[0]
goal_dim = env.observation_space.spaces["achieved_goal"].shape[0]
hidden_dim = [args.hidden_dim] * 2

writer = None

squasher = TanhSquasher()

sac = SAC(action_dim=action_dim, state_dim=state_dim + goal_dim, hidden_dim=hidden_dim,
          discount_factor=args.discount_factor, optimizer=Adam,
          writer=writer, scale_reward=args.scale_reward, reparam=args.reparam, deterministic=args.deterministic,
          target_update_interval=args.target_update_interval, lr=args.lr, soft_update_tau=args.soft_update_tau,
          td3_update_interval=args.td3_update_interval, squasher=squasher, weight_decay=args.weight_decay,
          grad_clip=args.grad_clip, loss_clip=args.loss_clip, clip_val_grad=args.clip_val_grad,
          clip_val_loss=args.clip_val_loss)

actor_path = f"model/{args.env_name}/actor_periodic.pt"
sac.load_model(actor_path=actor_path)

detertministic_eval = False
for i in range(20):
    reset_obs = env.reset()
    state = torch.Tensor(reset_obs["observation"])
    desired_goal = torch.Tensor(reset_obs["desired_goal"])
    done = False
    timestep = 0

    success = False
    while not done and timestep <= args.max_time_steps:
        env.render()
        action = sac.get_action(torch.cat([state, desired_goal]), deterministic=detertministic_eval).detach()
        observation, reward, done, info = gym_torchify(env.step(action.numpy()), is_goal_env=True)
        state = observation["observation"]
        timestep += 1
