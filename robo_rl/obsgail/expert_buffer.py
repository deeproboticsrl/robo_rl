import numpy as np
import pickle


class ExpertBuffer():
    """Unlike traditional replay buffers, this is a collection of
    trajectories not individual transitions"""

    def __init__(self, size=1000):
        self.size = size
        self.expert_buffer = []
        self.current_size = 0
        self.next_index = 0

    def sample(self,sample_size=1):
        """Sample a trajectory from the buffer in an uniformly random manner"""
        indices = np.random.randint(low=0, high=self.current_size, size=sample_size)
        return [self.expert_buffer[index] for index in indices]

    def add(self, trajectory):
        if self.current_size < self.size:
            self.current_size += 1
            self.expert_buffer.append(None)
        self.expert_buffer[self.next_index] = np.array(trajectory)
        # cyclic queue
        self.next_index = ( self.next_index + 1 ) % self.size

    def add_from_file(self, expert_file_path):
        with open(expert_file_path, "rb") as expert_file:
            """assumed file to contain an array/list of trajectories"""
            trajectories = np.array(pickle.load(expert_file))
            for trajectory in trajectories:
                self.add(trajectory)
