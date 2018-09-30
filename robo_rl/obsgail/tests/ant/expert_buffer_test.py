from robo_rl.obsgail.expert_buffer import ExpertBuffer
import robo_rl.common.utils.utils as utils
from robo_rl.obsgail.gail_env.ant_gail_env import AntGAILEnv

expert_buffer = ExpertBuffer()

expert_buffer.add_from_file("../../experts/ant_obs_forward_expert.pkl")

utils.print_heading("Size of expert buffer after adding trajectory(ies)")
print(expert_buffer.current_size)

expert_trajectories = expert_buffer.sample(1)
expert_trajectory = expert_trajectories[0]
ant_gail_env = AntGAILEnv()
# ant_gail_env.play_expert(expert_trajectory)

import numpy as np
for i in range(len(expert_trajectory)):
    """observations for qpos are got using self.sim.data.qpos.flat where qpos has length 15"""
    new_state_qpos = [0]*15
    new_state_qpos[9] = 0.1
    new_state_qpos[10] = 0.5
    """qvel in ant is of length 14 and concatenated after qpos"""
    new_state_qvel = [0]*14
    ant_gail_env.set_state(np.array(new_state_qpos), np.array(new_state_qvel))
    ant_gail_env.render()