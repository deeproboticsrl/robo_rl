from robo_rl.obsvail.gail_env.ant_gail_env import AntGAILEnv
import pickle
import robo_rl.common.utils.utils as utils

with open("../../experts/ant_obs_forward_expert.pkl", "rb") as expert_file:
    expert_trajectories = pickle.load(expert_file)

expert_trajectory = expert_trajectories[0]
utils.print_heading("Length of expert trajectory")
print(len(expert_trajectory))

utils.print_heading("Length of 1 observation")
print(len(expert_trajectory[0]))

ant_gail_env = AntGAILEnv()

utils.print_heading("Size of qpos and qvel")
print(len(ant_gail_env.sim.data.qpos.flat[2:]), len(ant_gail_env.sim.data.qvel.flat))

utils.print_heading("Feature vector and it's size")
feature_vector = ant_gail_env.partial_feature_mapper(expert_trajectory[10])
print(feature_vector, len(feature_vector))

ant_gail_env.play_expert(expert_trajectory)
