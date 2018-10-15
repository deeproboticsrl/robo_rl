import os

import gym
import numpy as np
import torch
from robo_rl.common import Buffer
from robo_rl.common.utils import gym_torchify, print_heading
from robo_rl.sac import SAC, TanhSquasher
from robo_rl.sac import get_sac_parser
from tensorboardX import SummaryWriter
from torch.optim import Adam

parser = get_sac_parser()
parser.add_argument('--env_name', default="FetchReach-v1", help="Should be GoalEnv")
args = parser.parse_args()

env = gym.make(args.env_name)

env.seed(args.env_seed)
torch.manual_seed(args.env_seed)
np.random.seed(args.env_seed)

action_dim = env.action_space.shape[0]
# state_dim = env.observation_space.spaces["observation"].shape[0]
state_dim = env.observation_space.spaces["achieved_goal"].shape[0]
goal_dim = env.observation_space.spaces["achieved_goal"].shape[0]
hidden_dim = [args.hidden_dim] * 2

unbiased = False
rewarding = False

if unbiased:
    logdir = "./tensorboard_log/unbiased_her"
else:
    logdir = "./tensorboard_log/biased_her"

# logdir += "dummy"

if rewarding:
    logdir += "_rewarding"
else:
    logdir += "_unrewarding"

deterministic_policy = False
deterministic_eval = True

if args.grad_clip:
    logdir += f"_grad_clip_{args.clip_val_grad}"
if args.loss_clip:
    logdir += f"_loss_clip_{args.clip_val_loss}"
if deterministic_eval:
    logdir += f"_deterministicTEST"
if deterministic_policy:
    logdir += f"_deterministicpolicy_"
logdir += f"_GOALIFIED_states_reward_scale={args.scale_reward}_tau={args.soft_update_tau}"
logdir += f"_samples={args.sample_batch_size}_hidden={hidden_dim}_discount_factor={args.discount_factor}"
logdir += f"_td3={args.td3_update_interval}_lr={args.lr}_weight_decay={args.weight_decay}"
logdir += f"_updates={args.updates_per_step}_num_episodes={args.num_episodes}"
logdir += f"_log_std_min={args.log_std_min}_max={args.log_std_max}_"

os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(log_dir=logdir)

squasher = TanhSquasher()

sac = SAC(action_dim=action_dim, state_dim=state_dim + goal_dim, hidden_dim=hidden_dim,
          discount_factor=args.discount_factor, optimizer=Adam,
          writer=writer, scale_reward=args.scale_reward, reparam=args.reparam, deterministic=args.deterministic,
          target_update_interval=args.target_update_interval, lr=args.lr, soft_update_tau=args.soft_update_tau,
          td3_update_interval=args.td3_update_interval, squasher=squasher, weight_decay=args.weight_decay,
          grad_clip=args.grad_clip, loss_clip=args.loss_clip, clip_val_grad=args.clip_val_grad,
          clip_val_loss=args.clip_val_loss, log_std_min=args.log_std_min, log_std_max=args.log_std_max)

buffer = Buffer(capacity=args.buffer_capacity)
rewards = []
update_count = 0
max_accuracy = -np.inf


def ld_to_dl(batch_list_of_dicts):
    batch_dict_of_lists = {}
    for k in batch_list_of_dicts[0].keys():
        batch_dict_of_lists[k] = []
    for dictionary in batch_list_of_dicts:
        for k in batch_list_of_dicts[0].keys():
            batch_dict_of_lists[k].append(dictionary[k])
    return batch_dict_of_lists


# num episodes after which to test
test_interval = 20
num_tests = 10

for cur_episode in range(1, args.num_episodes + 1):
    print(f"Starting episode {cur_episode}")

    reset_obs = env.reset()
    # state = torch.Tensor(reset_obs["observation"])
    state = torch.Tensor(reset_obs["achieved_goal"])
    desired_goal = torch.Tensor(reset_obs["desired_goal"])
    done = False
    timestep = 0
    episode_buffer = []

    while not done and timestep <= args.max_time_steps:
        episode_reward = 0
        action, log_prob = sac.get_action(torch.cat([state, desired_goal]), evaluate=True,
                                          deterministic=deterministic_policy)
        action = action.detach()
        observation, reward, done, _ = gym_torchify(env.step(action.numpy()), is_goal_env=True)
        reward = reward + 1
        next_state = observation["achieved_goal"]
        episode_buffer.append(dict(state=state, next_state=observation["achieved_goal"], action=action,
                                   done=done, achieved_goal=observation["achieved_goal"], log_prob=log_prob))
        sample = dict(state=torch.cat([state, desired_goal]), action=action, reward=reward,
                      next_state=torch.cat([next_state, desired_goal]), done=done)
        buffer.add(sample)

        episode_reward += reward
        state = next_state
        timestep += 1

    sac.writer.add_scalar("Policy linear layer 1 weight 0", sac.policy.linear_layers[0].weight[0][0], cur_episode)
    for name, param in sac.policy.named_parameters():
        sac.writer.add_histogram("policy_" + name, param.clone().cpu().data.numpy(), cur_episode)
    for name, param in sac.value.named_parameters():
        sac.writer.add_histogram("value_" + name, param.clone().cpu().data.numpy(), cur_episode)
    for name, param in sac.critics[0].named_parameters():
        sac.writer.add_histogram("critic1_" + name, param.clone().cpu().data.numpy(), cur_episode)
    for name, param in sac.critics[1].named_parameters():
        sac.writer.add_histogram("critic2_" + name, param.clone().cpu().data.numpy(), cur_episode)

    # add hindsight transitions
    final_goal = observation["achieved_goal"]
    for transition in episode_buffer:
        if rewarding:
            final_goal = transition["achieved_goal"]
        state = torch.cat([transition["state"], final_goal])
        log_prob_final_goal = sac.policy.compute_log_prob_action(state, sac.squasher, transition["action"]).detach()

        reward = env.compute_reward(achieved_goal=transition["achieved_goal"], desired_goal=final_goal, info=None)
        reward = reward + 1
        unbiased_reward = torch.Tensor(reward * np.exp(log_prob_final_goal) / np.exp(transition["log_prob"].detach()))

        if unbiased:
            reward = unbiased_reward
        done = transition["done"]
        if rewarding:
            done = torch.Tensor([True])
        hindisght_sample = dict(state=state,
                                action=transition["action"],
                                reward=torch.Tensor([reward]).detach(),
                                done=done,
                                next_state=torch.cat([transition["next_state"], final_goal]))
        buffer.add(hindisght_sample)

    if len(buffer) > args.sample_batch_size:
        for num_update in range(args.updates_per_step):
            update_count += 1
            batch_list_of_dicts = buffer.sample(batch_size=args.sample_batch_size)
            batch_dict_of_lists = ld_to_dl(batch_list_of_dicts)
            sac.policy_update(batch_dict_of_lists, update_number=update_count)

    if cur_episode % test_interval == 0:
        # test
        successes = []
        for i in range(num_tests):
            reset_obs = env.reset()
            state = torch.Tensor(reset_obs["achieved_goal"])
            desired_goal = torch.Tensor(reset_obs["desired_goal"])
            done = False
            timestep = 0

            success = False
            while not done and timestep <= args.max_time_steps:
                action = sac.get_action(torch.cat([state, desired_goal]), deterministic=deterministic_eval).detach()
                observation, reward, done, info = gym_torchify(env.step(action.numpy()), is_goal_env=True)
                state = observation["achieved_goal"]
                timestep += 1
                if 'is_success' in info:
                    success = info['is_success']
            successes.append(success)

        accuracy = sum(successes) / num_tests
        print_heading(f"Finished episode {cur_episode}")
        print(f"Accuracy {accuracy}")

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            # save current best model
            print(f"\nNew best model with accuracy {max_accuracy}")
            sac.save_model(actor_path=logdir, env_name=args.env_name, info='best')

        sac.writer.add_scalar("Accuracy ", accuracy, cur_episode / test_interval)

    if cur_episode % args.save_iter == 0:
        print(f"\nSaving periodically - iteration {cur_episode}")
        sac.save_model(actor_path=logdir, env_name=args.env_name, info="periodic")
        buffer.save_buffer(info=args.env_name)
