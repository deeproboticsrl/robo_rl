import torch
from robo_rl.common.utils.nn_utils import xavier_initialisation
from robo_rl.common.utils.utils import print_heading
from robo_rl.sac.gaussian_policy import GaussianPolicy
from robo_rl.sac.squasher import TanhSquasher, SigmoidSquasher

torch.manual_seed(0)

policy = GaussianPolicy(state_dim=2, action_dim=3, hidden_dim=[4, 5])
xavier_initialisation(policy)

state = torch.Tensor([0, 0.2])
mean, log_std = policy.forward(state)
print_heading("Mean and log_std")
print(mean, log_std)

tanh_squasher = TanhSquasher()
print_heading("Action and log_prob with tanh squasher")
actions, log_prob = policy.get_action(state, tanh_squasher,evaluate=True)
print(actions, log_prob)

sigmoid_squasher = SigmoidSquasher()
print_heading("Action and log_prob with sigmoid squasher")
actions, log_prob = policy.get_action(state, sigmoid_squasher, evaluate=True)
print(actions, log_prob)

tanh_squasher = TanhSquasher()
print_heading("Action and log_prob with tanh squasher deterministically")
actions, log_prob = policy.get_action(state, tanh_squasher, deterministic=True, evaluate=True)
print(actions, log_prob)
