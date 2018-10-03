import numpy as np
import copy


class ReplayBuffer:

    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, item):

        if len(self) > self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = copy.deepcopy(item)

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):

        if batch_size > len(self):
            raise ValueError('Sampling batch size greater than buffer size')

        indices = np.random.randint(0, len(self), batch_size)

        return self.buffer[indices]





