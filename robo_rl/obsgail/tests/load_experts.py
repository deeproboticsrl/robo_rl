import pickle

experts_file_path = "./expertsss.obs"
expert_trajectories = []

with open(experts_file_path, "rb") as f:
    expert_trajectories = pickle.load(f)

print(len(expert_trajectories))

