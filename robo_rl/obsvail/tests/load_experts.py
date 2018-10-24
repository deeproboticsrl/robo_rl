import pickle

experts_file_path = "../experts/sampled_experts.obs"

with open(experts_file_path, "rb") as f:
    expert_trajectories = pickle.load(f)

# (num_experts)
print(len(expert_trajectories))

# (trajectory_length)
print(len(expert_trajectories[0]["trajectory"]))

# (num_observations, 1)
print(expert_trajectories[0]["trajectory"][0]["state"].shape)
print(expert_trajectories[0]["context"])

# Should have 30 for each context - 2,3,4,5
context_bins = [0]*4
for expert_trajectory in expert_trajectories:
    context_decimal = expert_trajectory["context"][0] + 2 * expert_trajectory["context"][1]
    context_bins[context_decimal] += 1
print(context_bins)
