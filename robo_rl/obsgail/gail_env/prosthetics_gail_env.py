from robo_rl.obsgail.gail_env.gail_env import GAILEnv
from osim.env import ProstheticsEnv


class PrsotheticsGAILEnv(ProstheticsEnv, GAILEnv):

    def __init__(self, visualize = True, integrator_accuracy = 5e-5, difficulty=0, seed=0,
                 feature_mapper=None):
        ProstheticsEnv().__init__(self, visualize=visualize, integrator_accuracy=integrator_accuracy,
                                  difficulty=difficulty, seed=seed)
        GAILEnv.__init__(self, feature_mapper)

    def play_expert(self, expert_trajectory):
        pass
