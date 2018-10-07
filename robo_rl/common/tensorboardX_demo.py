import os

import torch
import torch.nn as nn
import torch.nn.functional as torchfunc
import torch.optim as optim
from robo_rl.common.networks.linear_network import LinearNetwork
from robo_rl.common.utils.nn_utils import xavier_initialisation
from robo_rl.common.utils.utils import print_heading
from tensorboardX import SummaryWriter

torch.manual_seed(0)

linear_network = LinearNetwork(layers_size=[1, 5, 5, 2],final_layer_function=torch.sigmoid,
                               activation_function=torchfunc.relu)

xavier_initialisation(linear_network)

mod_value = 4
lr = 0.01

x = torch.Tensor([i % mod_value for i in range(1000)]).reshape(1000, 1)
y = torch.LongTensor([[i % 2, (i + 1) % 2] for i in range(1000)]).reshape(1000, 2)

logdir = f"./tensorboard_log/mod={mod_value}_lr={lr}"
os.makedirs(logdir, exist_ok=True)

writer = SummaryWriter(log_dir=logdir)

writer.add_graph(linear_network, x[0])

for num_iteration in range(1000):
    y_pred = linear_network.forward(x, )

    a = y_pred.unsqueeze(1)
    y_not_pred = 1 - y_pred
    a_not_pred = y_not_pred.unsqueeze(1)

    y_target = torch.cat([a_not_pred, a], 1)

    loss = nn.CrossEntropyLoss()
    error = loss(input=y_target, target=y)

    optimizer = optim.Adam(linear_network.parameters(), lr=lr)

    optimizer.zero_grad()
    error.backward()
    optimizer.step()

    writer.add_scalar("data/error", error, num_iteration)

    y_hat = torch.round(y_pred).long()
    num_wrong = torch.sum(torch.abs(y_hat - y)).float()

    accuracy = 1 - (num_wrong / y.shape[0] / y.shape[1])

    # print(y[0], y_hat[0])

    writer.add_scalar("data/accuracy", accuracy, num_iteration)

    for name, param in linear_network.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), num_iteration)

    writer.add_histogram("y_pred/probability", y_pred, num_iteration)
    writer.add_histogram("y_pred/prediction", y_hat, num_iteration)

print_heading("Final loss")
print(error.detach())

print_heading("Final Accuracy")
print(accuracy.detach())

writer.close()
