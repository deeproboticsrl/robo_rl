import pickle

import numpy as np
from robo_rl.common.buffer.buffer import Buffer


class TrajectoryBuffer(Buffer):
    """Unlike traditional replay buffers, this is a collection of
    trajectories not individual transitions"""

    def __init__(self, capacity=1000):
        super().__init__(capacity)

    def sample_timestep(self, batch_size, timestep):
        """Sample from trajectories at a particular timestep"""
        if batch_size > len(self):
            raise ValueError('Sampling batch size greater than buffer size')

        trajectory_indices = np.random.randint(0, len(self), batch_size)
        return [{"context": self.buffer[trajectory_index]["context"],
                 "transition": self.buffer[trajectory_index]["trajectory"][timestep]}
                for trajectory_index in trajectory_indices]

    def add_from_file(self, expert_file_path):
        with open(expert_file_path, "rb") as expert_file:
            """assumed file to contain an array/list of trajectories"""
            trajectories = np.array(pickle.load(expert_file))
            for trajectory in trajectories:
                self.add(trajectory)
