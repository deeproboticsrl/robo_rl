from robo_rl.obsgail.gail_env.gail_env import GAILEnv
from gym.envs.mujoco.ant import AntEnv
import numpy as np


class AntGAILEnv(AntEnv, GAILEnv):

    def __init__(self, feature_mapper=None):
        AntEnv.__init__(self)
        GAILEnv.__init__(self, feature_mapper)

    @staticmethod
    def partial_feature_mapper(state, context=None):
        """returns rotational position and velocity for joint of the 4 legs and translational for torso"""
        features = []
        # torso qpos
        features.extend(np.array(state[0:2]))
        # legs qpos
        features.extend(np.array(state[7:15]))
        # torso qvel
        features.extend(np.array(state[16:18]))
        # legs qvel
        features.extend(np.array(state[23:31]))
        return features

    def play_expert(self, expert_trajectory):
        """Expert trajectory is a list of observations"""
        for i in range(len(expert_trajectory)):
            """observations for qpos are got using self.sim.data.qpos.flat where qpos has length 15"""
            new_state_qpos = expert_trajectory[i][0:15]
            """qvel in ant is of length 14 and concatenated after qpos"""
            new_state_qvel = expert_trajectory[i][16:30]
            self.set_state(np.array(new_state_qpos),np.array(new_state_qvel))
            self.render()

    """override since want to use torso positions too"""
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
