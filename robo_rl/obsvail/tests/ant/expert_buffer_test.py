from robo_rl.obsvail.expert_buffer import ExpertBuffer
import robo_rl.common.utils.utils as utils
from robo_rl.obsvail.gail_env.ant_gail_env import AntGAILEnv

expert_buffer = ExpertBuffer()

expert_buffer.add_from_file("../../experts/ant_obs_forward_expert.pkl")

utils.print_heading("Size of expert buffer after adding trajectory(ies)")
print(len(expert_buffer))

expert_trajectories = expert_buffer.sample(1)
expert_trajectory = expert_trajectories[0]
ant_gail_env = AntGAILEnv()
ant_gail_env.play_expert(expert_trajectory)
