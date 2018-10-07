from abc import ABC, abstractmethod
import torch


class Squasher(ABC):

    @abstractmethod
    def squash(self, x):
        pass

    def derivative(self, x):
        pass


class TanhSquasher(Squasher):

    def squash(self, x):
        return torch.tanh(x)

    def derivative(self, x):
        return 1 - torch.tanh(x)**2

class SigmoidSquasher(Squasher):

    def squash(self, x):
        return torch.sigmoid(x)

    def derivative(self, x):
        return torch.sigmoid(x)*(1 - torch.sigmoid(x))
