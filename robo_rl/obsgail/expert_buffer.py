import numpy as np
import pickle
from robo_rl.common.replaybuffer import ReplayBuffer


class ExpertBuffer(ReplayBuffer):
    """Unlike traditional replay buffers, this is a collection of
    trajectories not individual transitions"""

    def __init__(self, capacity=1000):
        super().__init__(capacity)

    def add_from_file(self, expert_file_path):
        with open(expert_file_path, "rb") as expert_file:
            """assumed file to contain an array/list of trajectories"""
            trajectories = np.array(pickle.load(expert_file))
            for trajectory in trajectories:
                self.add(trajectory)

