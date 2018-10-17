import torch
import torch.nn.functional as torchfunc
from robo_rl.common import LinearPFNN
from robo_rl.common import print_heading
from torch.optim import Adam

torch.manual_seed(0)

layers_size = [1, 2, 1]
final_layer_function = torch.sigmoid
activation_function = torchfunc.elu
num_networks = 3

pfnn = LinearPFNN(layers_size=layers_size, final_layer_function=final_layer_function,
                  activation_function=activation_function, num_networks=num_networks)

input_tensor = torch.Tensor([0.291])
phase = 0.991

x = {"input": input_tensor, "phase": phase}

input_tensor = x["input"]

print_heading("Phase and corresponding indices")
phase = x["phase"]
# Enforce phase in [0,1)
phase = phase - int(phase)
print("Phase".ljust(25), phase)

# Get indices for interval endpoints
left_index = int(phase * num_networks) % num_networks
right_index = (left_index + 1) % num_networks
left_phase = left_index / num_networks
right_phase = left_phase + (1 / num_networks)

# phase = weight * left_phase + (1-weight) * right_phase
weight = (right_phase - phase) * num_networks

print("Num Networks".ljust(25), num_networks)
print("Left Index".ljust(25), left_index)
print("Right Index".ljust(25), right_index)
print("Left phase".ljust(25), left_phase)
print("Right phase".ljust(25), right_phase)
print("Interpolation Weight".ljust(25), weight)

optimiser = Adam(pfnn.basis_networks.parameters(), lr=0.01)

left_net = pfnn.basis_networks[left_index]
right_net = pfnn.basis_networks[right_index]

for main_param, left_param, right_param in zip(pfnn.main_network.parameters(), left_net.parameters(),
                                               right_net.parameters()):
    main_param.data = weight * left_param.data + (1 - weight) * right_param.data

y = pfnn.main_network.forward(input_tensor)
# y = right_net.forward(input_tensor)

print_heading("Output")
print("y".ljust(25), y)

print_heading("Network weights before backprop")
for i in range(num_networks):
    print(f"Network {i}")
    print(pfnn.basis_networks[i].state_dict()['linear_layers.0.weight'])

optimiser.zero_grad()
y.backward()
optimiser.step()

print_heading("Network weights after backprop")
for i in range(num_networks):
    print(f"Network {i}")
    print(pfnn.basis_networks[i].state_dict()['linear_layers.0.weight'])
