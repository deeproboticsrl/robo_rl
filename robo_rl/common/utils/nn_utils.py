import torch.nn as nn


# xavier initialization does random operation so calling it twice on same network will yield different results
def xavier_initialisation(*module):
    modules = [*module]
    for i in range(len(modules)):
        if type(modules[i]) in [nn.Linear]:
            nn.init.xavier_normal_(modules[i].weight.data)


def print_all_modules(network):
    for idx, module in enumerate(network.modules()):
        print(idx, '->', module)


def no_activation(x):
    return x
