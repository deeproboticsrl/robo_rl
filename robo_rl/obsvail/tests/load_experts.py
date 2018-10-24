import pickle

experts_file_path = "../experts/sampled_experts.obs"

with open(experts_file_path, "rb") as f:
    expert_trajectories = pickle.load(f)

# (num_experts)
print(len(expert_trajectories))

# (trajectory_length, num_observations, 1)
print(expert_trajectories[0][0]["state"].shape)
print(expert_trajectories[0][0]["context"])

# Should have 30 for each context - 2,3,4,5
context_bins = [0]*4
for expert_trajectory in expert_trajectories:
    context_bins[int(expert_trajectory[0]["context"])-2] += 1
print(context_bins)
