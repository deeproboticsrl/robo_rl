import torch.nn as nn


def xavier_initialisation(module):
    if type(module) in [nn.Linear]:
        nn.init.xavier_normal_(module.weight.data)


def print_all_modules(network):
    for idx, module in enumerate(network.modules()):
        print(idx, '->', module)

