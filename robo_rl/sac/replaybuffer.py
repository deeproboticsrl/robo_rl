import numpy as np


class ReplayBuffer:

    def __init__(self,size):
        self.size = size
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        """
        :params: transition at each time step

        """
        if len(self.buffer) > self.size:
            self.buffer.append(None)
        self.buffer[self.position] = [state, action, reward , next_state, done]

        self.position = (self.position +1) % self.size


    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            raise ValueError('batch_size greater than current buffer')
        l= len(self.buffer)
        indices = np.random.randint(0, l, batch_size)

        return dict(state=self.buffer[0][indices],
                    action=self.buffer[1][indices],
                    reward=self.buffer[2][indices],
                    next_state=self.buffer[3][indices],
                    done=self.buffer[4][indices])





