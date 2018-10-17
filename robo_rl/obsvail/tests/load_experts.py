import pickle

experts_file_path = "../experts/sampled_experts.obs"

with open(experts_file_path, "rb") as f:
    expert_trajectories = pickle.load(f)

# (num_experts, trajectory_length, num_observations, 1)
print(expert_trajectories.shape)
