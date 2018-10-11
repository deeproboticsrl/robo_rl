from abc import ABC, abstractmethod

import torch


class Squasher(ABC):

    @abstractmethod
    def squash(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass

    def inverse(self, x):
        pass


class TanhSquasher(Squasher):

    def squash(self, x):
        return torch.tanh(x)

    def derivative(self, x):
        return 1 - torch.tanh(x) ** 2

    def inverse(self, x):
        return 0.5 * (torch.log(1 + x) / (1 - x))


class GAAFTanhSquasher(Squasher):

    def squash(self, x):
        k = 10000
        g = (x * k - torch.floor(x * k) - 0.5) / k
        # if x>0:
        #     s = torch.exp(-x)
        # else:
        #     s = torch.exp(x)
        y = torch.tanh(x) + g
        return y

    def derivative(self, x):
        return 1 - torch.tanh(x) ** 2 + 1


class SigmoidSquasher(Squasher):

    def squash(self, x):
        return torch.sigmoid(x)

    def derivative(self, x):
        return torch.sigmoid(x) * (1 - torch.sigmoid(x))


class NoSquasher(Squasher):

    def squash(self, x):
        return x

    def derivative(self, x):
        return torch.Tensor([1])

    def inverse(self, x):
        return x
